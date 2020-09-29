import directories
import utils
from datasets import PairDataBuilder, MentionDataBuilder, DocumentDataBuilder
from word_vectors import WordVectors
import random
import numpy as np


def explore_pairwise_features():
    pos_sum, neg_sum = np.zeros(9), np.zeros(9)
    pos_count, neg_count = 0, 0
    for i, d in enumerate(utils.load_json_lines(directories.RAW + "train")):
        for key in d["labels"].keys():
            if d["labels"][key] == 1:
                pos_sum += d["pair_features"][key]
                pos_count += 1
            else:
                neg_sum += d["pair_features"][key]
                neg_count += 1
        print("positive counts", list(pos_sum))
        print("negative counts", list(neg_sum))
        print("feature odds", list(np.divide(pos_sum / pos_count,
                                             (pos_sum / pos_count + neg_sum / neg_count))))
        print()


def build_dataset(vectors, name, tune_fraction=0.0, reduced=False, columns=None):
    doc_vectors = utils.load_pickle(directories.MISC + name.replace("_reduced", "") +
                                   "_document_vectors.pkl")

    main_pairs = PairDataBuilder(columns)
    tune_pairs = PairDataBuilder(columns)
    main_mentions = MentionDataBuilder(columns)
    tune_mentions = MentionDataBuilder(columns)
    main_docs = DocumentDataBuilder(columns)
    tune_docs = DocumentDataBuilder(columns)

    print("Building dataset", name + ("/tune" if tune_fraction > 0 else ""))
    p = utils.Progbar(target=(2 if reduced else utils.lines_in_file(directories.RAW + name)))
    for i, d in enumerate(utils.load_json_lines(directories.RAW + name)):
        if reduced and i > 2:
            break
        p.update(i + 1)

        if reduced and tune_fraction != 0:
            pairs, mentions, docs = (main_pairs, main_mentions, main_docs) \
                if i == 0 else (tune_pairs, tune_mentions, tune_docs)
        else:
            pairs, mentions, docs = (main_pairs, main_mentions, main_docs) \
                if random.random() > tune_fraction else (tune_pairs, tune_mentions, tune_docs)

        ms, ps = mentions.size(), pairs.size()
        mention_positions = {}
        for mention_num in sorted(d["mentions"].keys(), key=int):
            mention_positions[mention_num] = mentions.size()
            mentions.add_mention(d["mentions"][mention_num], vectors,
                                 doc_vectors[d["mentions"][mention_num]["doc_id"]])

        for key in sorted(d["labels"].keys(), key=lambda k: (int(k.split()[1]), int(k.split()[0]))):
            k1, k2 = key.split()
            pairs.add_pair(d["labels"][key], mention_positions[k1], mention_positions[k2],
                           int(d["mentions"][k1]["doc_id"]),
                           int(d["mentions"][k1]["mention_id"]),
                           int(d["mentions"][k2]["mention_id"]),
                           d["pair_features"][key])

        me, pe = mentions.size(), pairs.size()
        docs.add_doc(ms, me, ps, pe, d["document_features"])

    suffix = ("_reduced" if reduced else "")
    if tune_mentions.size() > 0:
        tune_mentions.write(name + "_tune" + suffix)
        tune_pairs.write(name + "_tune" + suffix)
        tune_docs.write(name + "_tune" + suffix)
        main_mentions.write(name + "_train" + suffix)
        main_pairs.write(name + "_train" + suffix)
        main_docs.write(name + "_train" + suffix)
    else:
        main_mentions.write(name + suffix)
        main_pairs.write(name + suffix)
        main_docs.write(name + suffix)


def build_datasets(reduced=False, columns=None):
    random.seed(0)
    vectors = WordVectors(load=True)
    build_dataset(vectors, "train", reduced=reduced, columns=columns, tune_fraction=0.15)
    build_dataset(vectors, "train", reduced=reduced, columns=columns)
    build_dataset(vectors, "dev", reduced=reduced, columns=columns)
    build_dataset(vectors, "test", reduced=reduced, columns=columns)


if __name__ == '__main__':
    build_datasets(reduced=True)
    build_datasets(reduced=False)
