import argparse
import numpy as np

import directories
import utils
import pairwise_models
import clustering_models
import word_vectors

"""
Exports a model into a form readable by CoreNLP.
"""

def write_word_vectors(model, weights_name, path):
    w = word_vectors.WordVectors(load=True)
    w.vectors = np.asarray(pairwise_models.get_weights(model, weights_name)[0])
    write_vectors(w, path + 'vectors_learned')

    w = word_vectors.WordVectors(keep_all_words=True)
    write_vectors(w, path + 'vectors_pretrained_all')


def write_weights(model, weights_name, path):
    weights = pairwise_models.get_weights(model, weights_name)

    w_ana = clustering_models.anaphoricity_weights(weights)
    write_matrices(w_ana, path + 'anaphoricity_weights')

    w_pair = clustering_models.pair_weights(weights)
    first = w_pair[0]
    s = 832 if directories.CHINESE else 650
    write_matrices([first[:s, :], first[s:2 * s, :], first[2 * s:, :]] + w_pair[1:],
                   path + 'pairwise_weights')


def write_vectors(vectors, path):
    with open(path, 'wb') as f:
        for w, i in vectors.vocabulary.items():
            if w == word_vectors.UNKNOWN_TOKEN:
                w = "*UNK*"
            f.write((w + " " + " ".join(map(str, vectors.vectors[i])) + "\n").encode('utf-8'))

def write_matrices(ms, fname):
    print("Writing matrices to " + fname)
    print([m.shape for m in ms])
    with open(fname, 'w') as f:
        for m in ms:
            if len(m.shape) == 1:
                f.write(" ".join(map(str, m)) + "\n")
            else:
                for i in range(m.shape[0]):
                    f.write(" ".join(map(str, m[i])) + "\n")
            f.write("\n\n")

def parse_args():
    parser = argparse.ArgumentParser(description='Exports a model to text for use elsewhere, such as corenlp')
    parser.add_argument('--model_name', default='reward_rescaling',
                        help='Name of the model to export')
    parser.add_argument('--weights_name', default='final_weights',
                        help='Name of the weights to export')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    weights_name = args.weights_name
    model_name = args.model_name

    path = directories.MODELS + model_name + "/exported_weights/"
    utils.mkdir(path)
    write_word_vectors(model_name, weights_name, path)
    write_weights(model_name, weights_name, path)
