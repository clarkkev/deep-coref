import directories
import util
import model_properties
import evaluation
from document import Document

import numpy as np
import timer
import time
import dataset
import pairwise_models
import shutil
import os

from collections import defaultdict
from pprint import pprint
from sklearn.metrics import accuracy_score, average_precision_score, precision_score,\
                            recall_score

EVAL_THRESHOLDS = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]


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

    def finish(self):
        loss = self.loss_sum / self.n_examples
        if not self.model_props.anaphoricity:
            printout = "{:} = loss: {:.4f} - P@1: {:}/{:} = {:.2f}%"\
                .format(self.name, loss, self.CL, self.CL + self.WL,
                        100 * self.CL / float(self.CL + self.WL))
        else:
            ana_prec = self.CN / max(1, float(self.CN + self.FN))
            ana_rec = self.CN / max(1, float(self.CN + self.FL))
            printout = "{:} = loss: {:.4f} - CN: {:} - CL: {:} - FN: {:} - FL: {:} - WL: {:}\n" \
                       "      ranking: {:.4f} - anaphoricity: {:.4f}"\
                .format(self.name, loss, self.CN, self.CL, self.FN, self.FL, self.WL,
                        self.CL / max(1, float(self.CL + self.WL)),
                        2 * ana_prec * ana_rec / max(1e-6, ana_prec + ana_rec))
        result = {
                self.name + " loss": loss,
                self.name + " CN": self.CN,
                self.name + " CL": self.CL,
                self.name + " FN": self.FN,
                self.name + " FL": self.FL,
                self.name + " WL": self.WL,
        }
        return printout, result


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

    def finish(self):
        self.y_pred = np.array(self.y_pred, dtype='float32')
        self.y_true = np.array(self.y_true)
        auc = average_precision_score(self.y_true, self.y_pred)
        loss = self.loss_sum / self.n_examples

        metrics = {thresh: self.get_metrics(thresh) for thresh in EVAL_THRESHOLDS}
        best_accuracy = max(m['accuracy'] for m in metrics.values())
        best_f1_threshold = max(metrics.keys(), key=lambda t: metrics[t]['f1'])
        result = {
            self.name + " loss": loss,
            self.name + " auc": auc,
            self.name + " best_threshold": best_f1_threshold
        }
        result.update({self.name + " " + k: v for k, v in metrics[0.5]
                      .iteritems()})
        result.update({self.name + " best_" + k: v for k, v in
                       metrics[best_f1_threshold].iteritems()})
        result[self.name + " best_accuracy"] = best_accuracy
        printout = "{:} - loss: {:.4f} - auc: {:.4f} - f1: {:.4f} (thresh={:.2f})".format(
                self.name, loss, auc, metrics[best_f1_threshold]['f1'], best_f1_threshold)
        return printout, result

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


def update_doc(doc, X, scores, links_record=None, scores_record=None):
    s = scores[1][:, 0]
    starts_ends = zip(X['starts'][:, 0], X['ends'][:, 0])
    for (start, end) in starts_ends:
        action_scores = s[start:end]
        link = np.argmax(action_scores)
        m1, m2 = X['ids'][start + link]

        if links_record is not None:
            for pair, link_score in zip(X['ids'][start:end], action_scores):
                scores_record[doc.did][tuple(pair)] = link_score
            if m1 != -1:
                links_record[doc.did].append((m1, m2))
        doc.link(m1, m2)


def run_over_docs(dataset_name, docs, model):
    docs_by_id = {doc.did: doc for doc in docs}
    prog = util.Progbar(dataset_name.n_batches)
    for i, X in enumerate(dataset_name):
        if X['y'].size == 0:
            continue
        scores = model.predict_on_batch(X)
        update_doc(docs_by_id[X['did']], X, scores)
        prog.update(i + 1)


def evaluate(docs, prefix):
    results = {}
    for name, metric in [(' muc', evaluation.muc), (' b3', evaluation.b_cubed),
                         (' ceafe', evaluation.ceafe)]:
        p, r, f1 = evaluation.evaluate_documents(docs, metric)
        results[prefix + name] = f1
        results[prefix + name + ' precision'] = p
        results[prefix + name + ' recall'] = r
    muc, b3, ceafe = results[prefix + ' muc'], results[prefix + ' b3'], results[prefix + ' ceafe']
    conll = (muc + b3 + ceafe) / 3
    print "{:} - MUC: {:0.2f} - B3: {:0.2f} - CEAFE: {:0.2f} - CoNLL {:0.2f}".format(
        prefix, 100 * muc, 100 * b3, 100 * ceafe, 100 * conll)
    results[prefix + ' conll'] = conll
    return results


def main(n_epochs=10000, test_only=False, write_scores=False, reduced=False,
         model_props=None, validate='dev'):
    if model_props is None:
        model_props = model_properties.MentionRankingProps()
    if model_props.ranking or \
            model_props.top_pairs:
        write_start = 0
        write_every = 10
    else:
        write_start = 80
        write_every = 20

    print "Training", directories.MODEL
    pprint(model_props.__dict__)

    if not test_only:
        model_props.write(directories.MODEL + 'model_props.pkl')
        util.rmkdir(directories.MODEL + 'src')
        for fname in os.listdir('.'):
            if fname.endswith('.py'):
                shutil.copyfile(fname, directories.MODEL + 'src/' + fname)

    print "Loading data"
    vectors = np.load(directories.RELEVANT_VECTORS + 'word_vectors.npy')
    if not test_only:
        train = dataset.DocumentBatchedDataset("train_reduced" if reduced else "train", model_props,
                                   max_pairs=10000, with_ids=True)
    tune = dataset.DocumentBatchedDataset(validate + "_reduced" if reduced else validate, model_props,
                              with_ids=True)

    print "Building model"
    model, opt = pairwise_models.get_model(tune, vectors, model_props)
    json_string = model.to_json()
    open(directories.MODEL + 'architecture.json', 'w').write(json_string)

    print "Training"
    print
    best_val_score = 1000
    best_val_score_in_window = 1000
    history = []
    for epoch in range(n_epochs):
        timer.start("train")
        print "EPOCH {:}, model = {:}".format((epoch + 1), directories.MODEL)
        if model_props.ranking:
            if not test_only:
                train_docs = util.load_pickle(directories.DOCUMENTS + 'train_docs.pkl')
            dev_docs = util.load_pickle(directories.DOCUMENTS + validate + '_docs.pkl')
            if reduced:
                dev_docs = dev_docs[:3]

        epoch_stats = {}
        if not test_only:
            model_weights = model.get_weights()
            if model_props.ranking:
                print "Running over train"
                run_over_docs(train, train_docs, model)
                epoch_stats.update(evaluate(train_docs, "train"))

            print "Training"
            prog = util.Progbar(train.n_batches)
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

        print "Testing"
        prog = util.Progbar(tune.n_batches)
        mt = RankingMetricsTracker("dev", model_props=model_props) \
            if model_props.ranking else ClassificationMetricsTracker("dev")
        mta = ClassificationMetricsTracker("dev anaphoricity", anaphoricity=True)

        docs_by_id = {doc.did: doc for doc in dev_docs} if model_props.ranking else {}
        links_record, scores_record = (defaultdict(list), defaultdict(dict)) \
            if write_scores else (None, None)
        for i, X in enumerate(tune):
            if X['y'].size == 0:
                continue
            progress = []
            scores = model.predict_on_batch(X)
            if model_props.ranking:
                update_doc(docs_by_id[X['did']], X, scores,
                           links_record=links_record, scores_record=scores_record)
            if model_props.anaphoricity and not model_props.ranking:
                progress.append(("anaphoricity loss", mta.update(X, scores[0][:, 0])))
            if not model_props.anaphoricity_only:
                progress.append(("loss", mt.update(
                    X, scores if model_props.ranking else
                    scores[1 if model_props.anaphoricity else 0][:, 0])))
            prog.update(i + 1, exact=progress)

        if write_scores:
            print "Writing scores"
            util.write_pickle(links_record, directories.MODEL + validate + '_links.pkl')
            util.write_pickle(scores_record, directories.MODEL + validate + '_scores.pkl')

        timer.start("metrics")
        if model_props.ranking:
            epoch_stats.update(evaluate(dev_docs, "dev"))
        epoch_stats["validate time"] = time.time() - prog.start
        if model_props.anaphoricity and not model_props.ranking:
            printout, result_a = mta.finish()
            epoch_stats.update(result_a)
            print printout
        if not model_props.anaphoricity_only:
            printout, result = mt.finish()
            epoch_stats.update(result)
            print printout
        timer.stop("metrics")

        if test_only:
            print " & ".join(map(lambda x: "{:.2f}".format(x * 100), [
                epoch_stats["dev muc precision"],
                epoch_stats["dev muc recall"],
                epoch_stats["dev muc"],
                epoch_stats["dev b3 precision"],
                epoch_stats["dev b3 recall"],
                epoch_stats["dev b3"],
                epoch_stats["dev ceafe precision"],
                epoch_stats["dev ceafe recall"],
                epoch_stats["dev ceafe"],
                epoch_stats["dev conll"],
            ]))
            return
        history.append(epoch_stats)
        util.write_pickle(history, directories.MODEL + 'history.pkl')
        score = -epoch_stats["dev conll"] if model_props.ranking else \
            (result["dev loss"] if not model_props.anaphoricity_only else
             result_a["dev anaphoricity loss"])
        if score < best_val_score:
            best_val_score = score
            print "New best {:}, saving model".format(
                "F1" if model_props.ranking else "validation loss")
            model.save_weights(directories.MODEL + "best_weights.hdf5", overwrite=True)
        if score < best_val_score_in_window and epoch > write_start:
            print "Best in last {:}, saved to weights_{:}".format(
                write_every, write_every * (epoch / write_every))
            best_val_score_in_window = score
            model.save_weights(directories.MODEL + "weights_{:}.hdf5".format(
                write_every * (epoch / write_every)), overwrite=True)
        if epoch % write_every == 0:
            best_val_score_in_window = 1000

        timer.stop("train")
        timer.print_totals()

        print


if __name__ == '__main__':
    main(test_only=True, validate='test', reduced=False, write_scores=True)
