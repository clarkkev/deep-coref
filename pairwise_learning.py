import directories
import utils
import model_properties
import evaluation
from document import Document

import numpy as np
import timer
import time
import datasets
import pairwise_models
import shutil
import os

from collections import defaultdict
from pprint import pprint
from sklearn.metrics import accuracy_score, average_precision_score, precision_score,\
                            recall_score


class RankingMetricsTracker:
    def __init__(self, name, model_props):
        self.name = name
        self.model_props = model_props
        self.loss_sum, self.n_examples = 0, 0
        self.CN, self.CL, self.FN, self.FL, self.WL, = 0, 0, 0, 0, 0

    def update(self, X, scores):
        y = scores[0][:, 0]
        self.loss_sum += y.sum()
        self.n_examples += y.size

        if len(scores) > 0:
            s = scores[1][:, 0]
            starts_ends = zip(X['starts'][:, 0], X['ends'][:, 0])
            for i, (start, end) in enumerate(starts_ends):
                link = np.argmax(s[start:end])
                costs = X["costs"][:, 0][start:end]
                c = costs[link]
                linked_new = link == end - start - 1
                correct_new = costs[end - start - 1] == 0

                if not self.model_props.anaphoricity:
                    if c == 0:
                        self.CL += 1
                    else:
                        self.WL += 1
                else:
                    if c == 0:
                        if linked_new:
                            self.CL += 1
                        else:
                            self.CN += 1
                    elif linked_new and not correct_new:
                        self.FN += 1
                    elif not linked_new and correct_new:
                        self.FL += 1
                    else:
                        self.WL += 1

        return self.loss_sum / self.n_examples

    def finish(self, stats):
        loss = self.loss_sum / self.n_examples
        if not self.model_props.anaphoricity:
            printout = "{:} - loss: {:.4f} - P@1: {:}/{:} = {:.2f}%"\
                .format(self.name, loss, self.CL, self.CL + self.WL,
                        100 * self.CL / float(self.CL + self.WL))
        else:
            ana_prec = self.CN / max(1, float(self.CN + self.FN))
            ana_rec = self.CN / max(1, float(self.CN + self.FL))
            printout = "{:} - loss: {:.4f} - CN: {:} - CL: {:} - FN: {:} - FL: {:} - WL: {:}\n" \
                       "      ranking: {:.4f} - anaphoricity: {:.4f}"\
                .format(self.name, loss, self.CN, self.CL, self.FN, self.FL, self.WL,
                        self.CL / max(1, float(self.CL + self.WL)),
                        2 * ana_prec * ana_rec / max(1e-6, ana_prec + ana_rec))
        stats.update({
                self.name + " loss": loss,
                self.name + " CN": self.CN,
                self.name + " CL": self.CL,
                self.name + " FN": self.FN,
                self.name + " FL": self.FL,
                self.name + " WL": self.WL,
        })
        print(printout)


class ClassificationMetricsTracker:
    def __init__(self, name, anaphoricity=False):
        self.name = name
        self.anaphoricity = anaphoricity
        self.y_pred, self.y_true = [], []
        self.loss_sum, self.n_examples = 0, 0

    def update(self, X, scores):
        y = X['anaphoricities'][:, 0] if self.anaphoricity else X['y'][:, 0]
        self.y_pred += list(scores)
        self.y_true += list(y)
        self.loss_sum += -np.sum(np.log(np.subtract(
            1, np.abs(y - np.clip(scores, 1.0e-7, 1 - 1.0e-7)))))
        self.n_examples += y.size
        return self.loss_sum / self.n_examples

    def finish(self, stats):
        self.y_pred = np.array(self.y_pred, dtype='float32')
        self.y_true = np.array(self.y_true)
        auc = average_precision_score(self.y_true, self.y_pred)
        loss = self.loss_sum / self.n_examples

        metrics = {thresh: self.get_metrics(thresh) for thresh in
                   [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]}
        best_accuracy = max(m['accuracy'] for m in metrics.values())
        best_f1_threshold = max(metrics.keys(), key=lambda t: metrics[t]['f1'])
        result = {
            self.name + " loss": loss,
            self.name + " auc": auc,
            self.name + " best_threshold": best_f1_threshold
        }
        result.update({self.name + " " + k: v for k, v in metrics[0.5]
                      .items()})
        result.update({self.name + " best_" + k: v for k, v in
                       metrics[best_f1_threshold].items()})
        result[self.name + " best_accuracy"] = best_accuracy

        stats.update(result)
        print("{:} - loss: {:.4f} - auc: {:.4f} - f1: {:.4f} (thresh={:.2f})".format(
                self.name, loss, auc, metrics[best_f1_threshold]['f1'], best_f1_threshold))

    def get_metrics(self, thresh):
        pred = np.clip(np.floor(self.y_pred / thresh), 0, 1)
        p, r = (0, 0) if pred.sum() == 0 else \
            (precision_score(self.y_true, pred), recall_score(self.y_true, pred))
        return {
            'accuracy': accuracy_score(self.y_true, pred),
            'precision': p,
            'recall': r,
            'f1': 0 if p == 0 or r == 0 else 2 * p * r / (p + r)
        }


def update_doc(doc, X, scores, saved_links=None, saved_scores=None):
    s = scores[1][:, 0]
    starts_ends = zip(X['starts'][:, 0], X['ends'][:, 0])
    for (start, end) in starts_ends:
        action_scores = s[start:end]
        link = np.argmax(action_scores)
        m1, m2 = X['ids'][start + link]

        if saved_links is not None:
            if m1 != -1:
                saved_links[doc.did].append((m1, m2))
        if saved_scores is not None:
            for pair, link_score in zip(X['ids'][start:end], action_scores):
                saved_scores[doc.did][tuple(pair)] = link_score

        doc.link(m1, m2)


def run_model_over_docs(dataset_name, docs, model):
    docs_by_id = {doc.did: doc for doc in docs}
    prog = utils.Progbar(dataset_name.n_batches)
    for i, X in enumerate(dataset_name):
        if X['y'].size == 0:
            continue
        scores = model.predict_on_batch(X)
        update_doc(docs_by_id[X['did']], X, scores)
        prog.update(i + 1)


def compute_metrics(docs, prefix):
    results = {}
    for name, metric in [(' muc', evaluation.muc), (' b3', evaluation.b_cubed),
                         (' ceafe', evaluation.ceafe), (' lea', evaluation.lea)]:
        p, r, f1 = evaluation.evaluate_documents(docs, metric)
        results[prefix + name] = f1
        results[prefix + name + ' precision'] = p
        results[prefix + name + ' recall'] = r
    muc, b3, ceafe, lea = \
        results[prefix + ' muc'], results[prefix + ' b3'], results[prefix + ' ceafe'], results[prefix + ' lea']
    conll = (muc + b3 + ceafe) / 3
    print("{:} - MUC: {:0.2f} - B3: {:0.2f} - CEAFE: {:0.2f} - LEA {:0.2f} - CoNLL {:0.2f}".format(
        prefix, 100 * muc, 100 * b3, 100 * ceafe, 100 * lea, 100 * conll))
    results[prefix + ' conll'] = conll
    return results


def set_costs(dataset, docs):
    docs_by_id = {doc.did: doc for doc in docs}
    prog = utils.Progbar(dataset.n_batches)
    for i, X in enumerate(dataset):
        if X['y'].size == 0:
            continue
        doc = docs_by_id[X['did']]
        doc_weight = (len(doc.mention_to_gold) + len(doc.mentions)) / 10.0
        for (start, end) in zip(X['starts'][:, 0], X['ends'][:, 0]):
            ids = X['ids'][start:end]
            ana = ids[0, 1]
            old_ant = doc.ana_to_ant[ana]
            doc.unlink(ana)
            costs = X['cost_ptrs'][start:end]
            for ant_ind in range(end - start):
                costs[ant_ind] = doc.link(ids[ant_ind, 0], ana, hypothetical=True, beta=1)
            doc.link(old_ant, ana)

            costs -= costs.max()
            costs *= -doc_weight
        prog.update(i + 1)


def test(model_props=None, model_name=None, weights_file='best_weights', dataset_name='test',
         save_output=True, save_scores=False):
    if model_props is None:
        model_props = model_properties.MentionRankingProps(name=model_name,
                                                           load_weights_from=model_name,
                                                           weights_file=weights_file)

    print("Loading data")
    vectors = np.load(directories.RELEVANT_VECTORS + 'word_vectors.npy')
    dataset = datasets.DocumentBatchedDataset(dataset_name, model_props, with_ids=True)
    docs = utils.load_pickle(directories.DOCUMENTS + dataset_name + '_docs.pkl')
    stats = {}

    print("Building model")
    model, _ = pairwise_models.get_model(dataset, vectors, model_props)

    print("Evaluating model on", dataset_name)
    evaluate_model(dataset, docs, model, model_props, stats,
                   save_output=save_output, save_scores=save_scores)
    timer.clear()
    utils.write_pickle(stats, model_props.path + dataset_name + "_results.pkl")


def evaluate_model(dataset, docs, model, model_props, stats, save_output=False, save_scores=False,
                   print_table=False):
    prog = utils.Progbar(dataset.n_batches)
    mt = RankingMetricsTracker(dataset.name, model_props=model_props) \
        if model_props.ranking else ClassificationMetricsTracker(dataset.name)
    mta = ClassificationMetricsTracker(dataset.name + " anaphoricity", anaphoricity=True)

    docs_by_id = {doc.did: doc for doc in docs} if model_props.ranking else {}
    saved_links, saved_scores = (defaultdict(list) if save_output else None,
                                 defaultdict(dict) if save_scores else None)
    for i, X in enumerate(dataset):
        if X['y'].size == 0:
            continue
        progress = []
        scores = model.predict_on_batch(X)
        if model_props.ranking:
            update_doc(docs_by_id[X['did']], X, scores,
                       saved_links=saved_links, saved_scores=saved_scores)
        if model_props.anaphoricity and not model_props.ranking:
            progress.append(("anaphoricity loss", mta.update(X, scores[0][:, 0])))
        if not model_props.anaphoricity_only:
            progress.append(("loss", mt.update(
                X, scores if model_props.ranking else
                scores[1 if model_props.anaphoricity else 0][:, 0])))
        prog.update(i + 1, exact=progress)

    if save_scores:
        print("Writing scores")
        utils.write_pickle(saved_scores, model_props.path + dataset.name + '_scores.pkl')
    if save_output:
        print("Writing output")
        utils.write_pickle(saved_links, model_props.path + dataset.name + '_links.pkl')
        utils.write_pickle(docs, model_props.path + dataset.name + '_processed_docs.pkl')

    timer.start("metrics")
    if model_props.ranking:
        stats.update(compute_metrics(docs, dataset.name))
    stats["validate time"] = time.time() - prog.start
    if model_props.anaphoricity and not model_props.ranking:
        mta.finish(stats)
    if not model_props.anaphoricity_only:
        mt.finish(stats)

    timer.stop("metrics")

    if print_table:
        print(" & ".join(map(lambda x: "{:.2f}".format(x * 100), [
            stats[dataset.name + " muc precision"],
            stats[dataset.name + " muc recall"],
            stats[dataset.name + " muc"],
            stats[dataset.name + " b3 precision"],
            stats[dataset.name + " b3 recall"],
            stats[dataset.name + " b3"],
            stats[dataset.name + " ceafe precision"],
            stats[dataset.name + " ceafe recall"],
            stats[dataset.name + " ceafe"],
            stats[dataset.name + " conll"],
        ])))


def train(model_props, n_epochs=10000, reduced=False, dev_set_name='dev'):
    print("Training", model_props.path)
    pprint(model_props.__dict__)

    model_props.write(model_props.path + 'model_props.pkl')
    utils.rmkdir(model_props.path + 'src')
    for fname in os.listdir('.'):
        if fname.endswith('.py'):
            shutil.copyfile(fname, model_props.path + 'src/' + fname)
    if model_props.ranking or \
            model_props.top_pairs:
        write_start = 0
        write_every = 10
    else:
        write_start = 80
        write_every = 20

    print("Loading data")
    vectors = np.load(directories.RELEVANT_VECTORS + 'word_vectors.npy')
    train = datasets.DocumentBatchedDataset("train_reduced" if reduced else "train",
                                            model_props, with_ids=True)
    dev = datasets.DocumentBatchedDataset(dev_set_name + "_reduced" if reduced else dev_set_name,
                                          model_props, with_ids=True)

    print("Building model")
    model, _ = pairwise_models.get_model(dev, vectors, model_props)
    json_string = model.to_json()
    open(model_props.path + 'architecture.json', 'w').write(json_string)

    best_val_score = 1000
    best_val_score_in_window = 1000
    history = []
    for epoch in range(n_epochs):
        timer.start("train")
        print("EPOCH {:}, model = {:}".format((epoch + 1), model_props.path))

        epoch_stats = {}
        model_weights = model.get_weights()
        train_docs = utils.load_pickle(directories.DOCUMENTS + 'train_docs.pkl')
        dev_docs = utils.load_pickle(directories.DOCUMENTS + dev_set_name + '_docs.pkl')
        if reduced:
            dev_docs = dev_docs[:3]

        if model_props.ranking:
            print("Running over training set")
            run_model_over_docs(train, train_docs, model)
            epoch_stats.update(compute_metrics(train_docs, "train"))
            if model_props.use_rewards:
                print("Setting costs")
                set_costs(train, train_docs)

        print("Training")
        prog = utils.Progbar(train.n_batches)
        train.shuffle()
        loss_sum, n_examples = 0, 0
        for i, X in enumerate(train):
            if X['y'].size == 0:
                continue
            batch_loss = model.train_on_batch(X)
            loss_sum += batch_loss * train.scale_factor
            n_examples += X['y'].size
            prog.update(i + 1, exact=[("train loss", loss_sum / n_examples)])
        epoch_stats["train time"] = time.time() - prog.start
        for k in prog.unique_values:
            epoch_stats[k] = prog.sum_values[k][0] / max(1, prog.sum_values[k][1])

        epoch_stats["weight diffs"] = [
            (np.sum(np.abs(new_weight - old_weight)), new_weight.size)
            for new_weight, old_weight in zip(model.get_weights(), model_weights)]
        summed = np.sum(map(np.array, epoch_stats["weight diffs"][1:]), axis=0)
        epoch_stats["total weight diff"] = tuple(summed)

        print("Testing on dev set")
        evaluate_model(dev, dev_docs, model, model_props, epoch_stats)

        history.append(epoch_stats)
        utils.write_pickle(history, model_props.path + 'history.pkl')
        score = -epoch_stats["dev conll"] if model_props.ranking else \
            (epoch_stats["dev loss"] if not model_props.anaphoricity_only else
             epoch_stats["dev anaphoricity loss"])
        if score < best_val_score:
            best_val_score = score
            print("New best {:}, saving model".format(
                "CoNLL F1" if model_props.ranking else "validation loss"))
            model.save_weights(model_props.path + "best_weights.hdf5", overwrite=True)
        if score < best_val_score_in_window and epoch > write_start:
            print("Best in last {:}, saved to weights_{:}".format(
                write_every, write_every * int(epoch / write_every)))
            best_val_score_in_window = score
            model.save_weights(model_props.path + "weights_{:}.hdf5".format(
                write_every * int(epoch / write_every)), overwrite=True)
            if epoch + write_every >= n_epochs:
                model.save_weights(model_props.path + "final_weights.hdf5", overwrite=True)
        if epoch % write_every == 0:
            best_val_score_in_window = 1000

        timer.stop("train")
        timer.print_totals()
        print()

    timer.clear()


if __name__ == '__main__':
    test(model_name='reward_rescaling', dataset_name='test', save_output=True, save_scores=False)
