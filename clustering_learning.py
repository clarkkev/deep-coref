import datasets
import clustering_models
import model_properties
import directories
import timer
import utils
import evaluation
from document import Document
from clustering_preprocessing import ActionSpace

import os
import random
import shutil
import numpy as np


class State:
    def __init__(self, doc, actionspace):
        self.list_idx = 0
        self.doc = doc
        self.actionspace = actionspace
        self._set_next_candidates()

    def _set_next_candidates(self):
        ana, ants = self.actionspace.actions[self.list_idx]
        self.current_mention = ana
        self.candidate_antecedents = []

        c2 = self.doc.mention_to_cluster[ana]
        seen = set()
        for ant in ants:
            c1 = self.doc.mention_to_cluster[ant]
            if c1 != c2 and c1 not in seen:
                seen.add(c1)
                self.candidate_antecedents.append(ant)

        if not self.candidate_antecedents:
            self.list_idx += 1
            if not self.is_complete():
                self._set_next_candidates()

    def get_f1(self):
        return self.doc.get_f1()

    def do_action(self, i):
        self.list_idx += 1
        if i < len(self.candidate_antecedents):
            self.doc.link(self.candidate_antecedents[i], self.current_mention)
        if not self.is_complete():
            self._set_next_candidates()

    def get_example(self, with_costs):
        X = {}
        pair_features = []
        starts, ends = [], []
        last_end = 0
        c2 = self.doc.mention_to_cluster[self.current_mention]
        for ant in self.candidate_antecedents:
            c1 = self.doc.mention_to_cluster[ant]
            for m1 in c1:
                for m2 in c2:
                    pair_features.append(self.actionspace.get_pair_features(m1, m2))
            starts.append(last_end)
            ends.append(len(pair_features))
            last_end = len(pair_features)

        X['pair_features'] = np.vstack(pair_features)
        X['mention_features'] = self.actionspace.get_mention_features(self.current_mention)
        X['starts'] = np.vstack(starts)
        X['ends'] = np.vstack(ends)
        if with_costs:
            X['costs'] = self.action_costs()
        return X

    def action_costs(self):
        timer.start("costs")
        costs = []
        for ant in self.candidate_antecedents:
            hypothetical_score = self.doc.link(ant, self.current_mention, hypothetical=True)
            costs.append(hypothetical_score)
        costs.append(self.get_f1())
        timer.stop("costs")
        costs = np.array(costs, dtype='float')
        costs -= costs.max()
        costs *= (len(self.doc.mention_to_gold) + len(self.doc.mentions)) / 100.0
        return -costs[:, np.newaxis]

    def is_complete(self):
        return self.list_idx == len(self.actionspace.actions)


class ReplayMemory:
    def __init__(self, trainer, model, replay_memory_size=1e10):
        self.size = 0
        self.memory = []
        self.trainer = trainer
        self.model = model
        self.replay_memory_size = replay_memory_size

    def update(self, example):
        self.memory.append(example)
        self.size += 1
        while self.size > self.replay_memory_size:
            self.train()

    def train(self):
        timer.start("train")
        X = self.memory.pop(int(random.random() * len(self.memory)))
        self.train_on_example(X)
        self.size -= 1
        timer.stop("train")

        if self.trainer.n == 1:
            print("Start training!")
            print()

    def train_all(self):
        timer.start("train")

        model_weights = self.model.get_weights()
        prog = utils.Progbar(len(self.memory))
        random.shuffle(self.memory)
        for i, X in enumerate(self.memory):
            loss = self.train_on_example(X)
            prog.update(i + 1, [("loss", loss)])
        self.size = 0
        self.memory = []
        timer.stop("train")
        weight_diffs = [
                (np.sum(np.abs(new_weight - old_weight)), new_weight.size)
                for new_weight, old_weight in zip(self.model.get_weights(), model_weights)]
        summed = np.sum(map(np.array, weight_diffs), axis=0)
        print("weight diffs", weight_diffs, summed)

    def train_on_example(self, X):
        loss = self.model.train_on_batch(X)
        self.trainer.n += 1
        return loss


class Aggregator:
    def __init__(self):
        self.total = 0
        self.count = 0

    def get_total(self):
        return self.total

    def get_avg(self):
        return 0 if self.count == 0 else self.total / self.count

    def update(self, value):
        self.total += value
        self.count += 1


class AgentRunner:
    def __init__(self, trainer, docs, data, message, replay_memory=None, beta=0,
                 docs_per_iteration=10000):
        self.trainer = trainer
        self.data = data
        self.model = trainer.model
        self.message = message
        self.replay_memory = replay_memory
        self.beta = beta
        self.loss_aggregator = Aggregator()
        self.evaluators = [
            evaluation.Evaluator(metric=evaluation.muc),
            evaluation.Evaluator(metric=evaluation.b_cubed),
            evaluation.Evaluator(metric=evaluation.ceafe)
        ]
        self.merged_pairs = {}
        self.training = self.replay_memory is not None

        print(self.message)
        random.shuffle(docs)
        if self.training:
            docs = docs[:docs_per_iteration]
        prog = utils.Progbar(len(docs))
        for i, (doc, actionstate) in enumerate(docs):
            self.trainer.doc = doc
            self.trainer.actionstate = actionstate

            if len(actionstate.possible_pairs) != 0:
                actionstate.load(self.data, self.trainer.pair_model,
                                 self.trainer.anaphoricity_model)
                s = State(doc, actionstate)
                doc_merged_pairs = self.run_agent(s, beta, i)
                for evaluator in self.evaluators:
                    evaluator.update(doc)
                self.merged_pairs[doc.did] = doc_merged_pairs
                doc.reset()
                actionstate.clear()

            muc, b3, ceafe = (self.evaluators[i].get_f1() for i in range(3))
            exact = [('muc', 100 * muc),
                     ('b3', 100 * b3),
                     ('ceafe', 100 * ceafe),
                     ('conll', 100 * (muc + b3 + ceafe) / 3),
                     ('loss', self.loss_aggregator.get_avg())]
            prog.update(i + 1, exact=exact)

    def run_agent(self, s, beta=0, iteration=1):
        timer.start("running agent")
        merged_pairs = []
        while not s.is_complete():
            example = s.get_example(self.training)
            n_candidates = example['starts'].size + 1

            if self.training:
                self.replay_memory.update(example)

            if random.random() > beta:
                if iteration == -1:
                    i = n_candidates - 1
                else:
                    timer.start("predict")
                    scores = self.model.predict_on_batch(example)[0]
                    if self.training:
                        self.loss_aggregator.update(np.sum(scores * example['costs']))
                    i = np.argmax(scores[:, 0])
                    timer.stop("predict")
            else:
                i = np.argmin(example['costs'][:, 0])
            if i != n_candidates - 1:
                merged_pairs.append((s.candidate_antecedents[i], s.current_mention))
            s.do_action(i)
        timer.stop("running agent")
        return merged_pairs


def evaluate(trainer, docs, data, message):
    ar = AgentRunner(trainer, docs, data, message)
    scores = {}
    for i, name in enumerate(['muc', 'b3', 'ceafe']):
        ev = ar.evaluators[i]
        p, r, f1 = ev.get_precision(), ev.get_recall(), ev.get_f1()
        scores[name] = f1
        scores[name + ' precision'] = p
        scores[name + ' recall'] = r
    scores['conll'] = (scores['muc'] + scores['b3'] + scores['ceafe']) / 3
    return scores, ar.loss_aggregator.get_avg(), ar.merged_pairs


def load_docs(dataset_name, word_vectors):
    return (datasets.Dataset(dataset_name, model_properties.MentionRankingProps(), word_vectors),
            zip(utils.load_pickle(directories.DOCUMENTS + dataset_name + '_docs.pkl'),
                utils.load_pickle(directories.ACTION_SPACE + dataset_name + '_action_space.pkl')))


class Trainer:
    def __init__(self, model_props, train_set='train', test_set='dev', n_epochs=200,
                 empty_buffer=True, betas=None, write_every=1, max_docs=10000):
        self.model_props = model_props
        if betas is None:
            betas = [0.8 ** i for i in range(1, 5)]
        self.write_every = write_every

        print("Model=" + model_props.path + ", ordering from " + directories.ACTION_SPACE)
        self.pair_model, self.anaphoricity_model, self.model, word_vectors = \
            clustering_models.get_models(model_props)
        json_string = self.model.to_json()
        open(model_props.path + 'architecture.json', 'w').write(json_string)
        utils.rmkdir(model_props.path + 'src')
        for fname in os.listdir('.'):
            if fname.endswith('.py'):
                shutil.copyfile(fname, model_props.path + 'src/' + fname)

        self.train_data, self.train_docs = load_docs(train_set, word_vectors)
        print("Train loaded")

        self.dev_data, self.dev_docs = self.train_data, self.train_docs
        print("Dev loaded!")

        self.test_data, self.test_docs = load_docs(test_set, word_vectors)
        print("Test loaded")

        random.seed(0)
        random.shuffle(self.train_docs)
        random.shuffle(self.dev_docs)
        random.shuffle(self.test_docs)
        self.train_docs = self.train_docs[:max_docs]
        self.dev_docs = self.dev_docs[:max_docs]
        self.test_docs = self.test_docs[:max_docs]

        self.epoch = 0
        self.n = 0
        self.history = []
        self.best_conll = 0
        self.best_conll_window = 0
        replay_memory = ReplayMemory(self, self.model)
        for self.epoch in range(n_epochs):
            print( 80 * "-")
            print("ITERATION", (self.epoch + 1), "model =", model_props.path)
            ar = AgentRunner(self, self.train_docs, self.train_data, "Training", replay_memory,
                             beta=0 if self.epoch >= len(betas) else betas[self.epoch])
            self.train_pairs = ar.merged_pairs
            if empty_buffer:
                replay_memory.train_all()
            self.run_evaluation()

    def run_evaluation(self):
        train_scores, train_loss, dev_pairs = evaluate(self, self.dev_docs, self.dev_data,
                                                       "Evaluating on train")
        test_scores, test_loss, test_pairs = evaluate(self, self.test_docs, self.test_data,
                                                      "Evaluating on test")
        epoch_stats = {
            "epoch": self.epoch,
            "n": self.n,
            "train_loss": train_loss,
            "test_loss": test_loss
        }
        epoch_stats.update({"train " + k: v for k, v in train_scores.iteritems()})
        epoch_stats.update({"test " + k: v for k, v in test_scores.iteritems()})
        self.history.append(epoch_stats)
        utils.write_pickle(self.history, self.model_props.path + 'history.pkl')
        timer.print_totals()

        test_conll = epoch_stats["test conll"]
        if self.epoch % self.write_every == 0:
            self.best_conll_window = 0
        if test_conll > self.best_conll:
            self.best_conll = test_conll
            print("New best CoNLL, saving model")
            self.save_progress(dev_pairs, test_pairs, "best")
        if test_conll > self.best_conll_window:
            self.best_conll_window = test_conll
            print("New best CoNLL in window, saving model")
            self.save_progress(dev_pairs, test_pairs,
                               str(self.write_every * int(self.epoch / self.write_every)))
        self.model.save_weights(self.model_props.path + "weights.hdf5", overwrite=True)

    def save_progress(self, dev_pairs, test_pairs, prefix):
        self.model.save_weights(self.model_props.path + prefix + "_weights.hdf5", overwrite=True)
        write_pairs(dev_pairs, self.model_props.path + prefix + "_dev_pairs")
        write_pairs(test_pairs, self.model_props.path + prefix + "_test_pairs")
        write_pairs(self.train_pairs, self.model_props.path + prefix + "_train_pairs")
        write_pairs(self.test_docs, self.model_props.path + prefix + "_test_processed_docs")


def write_pairs(pairs, path):
    with open(path, 'w') as f:
        for did, doc_merged_pairs in pairs.iteritems():
            f.write(str(did) + "\t")
            for m1, m2 in doc_merged_pairs:
                f.write(str(m1) + "," + str(m2) + " ")
            f.write("\n")


def main(model_props):
    Trainer(model_props)
