import utils
import directories
import word_vectors
import numpy as np
from collections import Counter


def docs(dataset_name):
    p = utils.Progbar(target=(utils.lines_in_file(directories.RAW + dataset_name)))
    for i, d in enumerate(utils.load_json_lines(directories.RAW + dataset_name)):
        p.update(i + 1)
        yield d


def write_words():
    words = Counter()
    for dataset_name in ["train", "dev", "test"]:
        inc = 1 if dataset_name == "train" else 0
        print("Adding words from", dataset_name)
        for d in docs(dataset_name):
            for mention in d["mentions"].values():
                for w in mention["sentence"]:
                    words[word_vectors.normalize(w)] += inc
                words[word_vectors.normalize(mention["dep_relation"])] += 1
    utils.write_pickle(words, directories.MISC + 'word_counts.pkl')


def write_document_vectors():
    vectors = word_vectors.WordVectors(load=True)
    for dataset_name in ["train", "dev", "test"]:
        print("Building document vectors for", dataset_name)
        doc_vectors = {}
        for d in docs(dataset_name):
            sentences = {}
            did = None
            for mention_num in sorted(d["mentions"].keys(), key=int):
                m = d["mentions"][mention_num]
                did = m["doc_id"]
                if m['sent_num'] not in sentences:
                    sentences[m['sent_num']] = m['sentence']

            v = np.zeros(vectors.vectors[0].size)
            n = 0
            for s in sentences.values():
                for w in s:
                    v += vectors.vectors[vectors[w]]
                    n += 1
            doc_vectors[did] = v / n
        utils.write_pickle(doc_vectors, directories.MISC + dataset_name + "_document_vectors.pkl")


def write_genres():
    sources = set()
    for dataset_name in ["train"]:
        print("Adding sources from", dataset_name)
        for d in docs(dataset_name):
            sources.add(d["document_features"]["source"])
    print(sources)
    utils.write_pickle({source: i for i, source in enumerate(sorted(sources))},
                      directories.MISC + 'genres.pkl')


def write_feature_names():
    raw_train = directories.RAW + 'train'
    try:
        utils.write_pickle({f: i for i, f in enumerate(next(
            utils.load_json_lines(raw_train))["pair_feature_names"])},
                           directories.MISC + 'pair_feature_names.pkl')
    except FileNotFoundError as e:
        if e.filename == raw_train:
            raise FileNotFoundError('Raw training data not found.  Perhaps you need to copy the original dataset first: %s' % e.filename) from e
        else:
            raise

def main():
    write_feature_names()
    write_genres()
    write_words()
    word_vectors.WordVectors().write(directories.RELEVANT_VECTORS)
    write_document_vectors()


if __name__ == '__main__':
    main()
