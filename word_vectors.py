import directories
import utils
import re
import numpy as np


ADD_WORD_THRESHOLD = 200
MISSING_TOKEN = '<missing>'
UNKNOWN_TOKEN = '<unk>'


def normalize(w):
    if w == "-LRB-":
        return "("
    elif w == "-RRB-":
        return ")"
    elif w == "-LCB-":
        return "{"
    elif w == "-RCB-":
        return "}"
    elif w == "-LSB-":
        return "["
    elif w == "-RSB-":
        return "]"
    return re.sub("\d", "0", w.lower())


class WordVectors:
    def __init__(self, load=False, vectors_file=directories.PRETRAINED_WORD_VECTORS,
                 keep_all_words=False):
        if load:
            self.vocabulary = utils.load_pickle(directories.RELEVANT_VECTORS + 'vocabulary.pkl')
            self.vectors = np.load(directories.RELEVANT_VECTORS + 'word_vectors.npy')
            self.d = self.vectors.shape[1]
        else:
            self.vocabulary = {}
            self.vectors = []
            word_counts = utils.load_pickle(directories.MISC + 'word_counts.pkl')
            with open(vectors_file, 'rb') as f:
                for line in f:
                    split = line.decode('utf8').split()
                    w = normalize(split[0])
                    if w not in self.vocabulary and (
                                        w == UNKNOWN_TOKEN or w in word_counts or keep_all_words):
                        vec = np.array(list(map(float, split[1:])), dtype='float32')
                        if not self.vectors:
                            self.d = vec.size
                            self.vectors.append(np.zeros(self.d))  # reserve 0 for mask
                        self.vocabulary[w] = len(self.vectors)
                        self.vectors.append(vec)

            n_unkowns = len([w for w in word_counts if w not in self.vocabulary])
            unknown_mass = sum(c for w, c in word_counts.items() if c < ADD_WORD_THRESHOLD and
                               w not in self.vocabulary)
            total_mass = sum(word_counts.values())
            print("Pretrained embedding size:", utils.lines_in_file(vectors_file))
            print("Unknowns by mass: {:}/{:} = {:.2f}%%"\
                .format(unknown_mass, total_mass, 100 * unknown_mass / float(total_mass)))
            print("Unknowns by count: {:}/{:} = {:.2f}%%"\
                .format(n_unkowns, len(word_counts), 100 * n_unkowns / float(len(word_counts))))

            for c, w in sorted([(w, c) for c, w in word_counts.items()], reverse=True):
                if w not in self.vocabulary and c > ADD_WORD_THRESHOLD:
                    print("Adding", w, "count =", c,
                    self.add_vector(w))
            if UNKNOWN_TOKEN not in self.vocabulary:
                print("No presupplied unknown token",
                self.add_vector(UNKNOWN_TOKEN))
            self.add_vector(MISSING_TOKEN)
        self.unknown = self.vocabulary[UNKNOWN_TOKEN]
        self.missing = self.vocabulary[MISSING_TOKEN]

    def __getitem__(self, w):
        w = normalize(w)
        if w in self.vocabulary:
            return self.vocabulary[w]
        return self.unknown

    def get(self, w):
        w = normalize(w)
        if w in self.vocabulary:
            return self.vocabulary[w]
        return self.add_vector(w)

    def add_vector(self, w):
        w = normalize(w)
        if w not in self.vocabulary:
            self.vocabulary[w] = len(self.vectors)
            self.vectors.append(np.zeros(self.d, dtype='float32'))
            return self.vocabulary[w]

    def write(self, path=directories.RELEVANT_VECTORS):
        np.save(path + 'word_vectors', np.vstack(self.vectors))
        utils.write_pickle(self.vocabulary, path + 'vocabulary.pkl')
