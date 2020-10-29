import utils
import directories
import shutil
import timer
import numpy as np
from collections import defaultdict

BATCH_SIZE = 8192
SCORE_THRESHOLD = -0.5 if directories.CHINESE else -2
MARGIN_THRESHOLD = -1 if directories.CHINESE else -1.5


class ActionSpace:
    def __init__(self, did, actions, possible_pairs):
        self.did = did
        self.actions = actions
        self.possible_pairs = possible_pairs
        self.mentions = [action[0] for action in actions]

    def load(self, data, pair_model, anaphoricity_model):
        timer.start("pair model")
        pair_features, self.pair_ids = data.vectorize_pairs(self.did, self.possible_pairs)
        self.pair_vectors = run_static_model(pair_features, pair_model)
        timer.stop("pair model")

        timer.start("anaphoricity model")
        mention_features, self.mention_ids = data.vectorize_mentions(self.did, self.mentions)
        self.mention_vectors = run_static_model(mention_features, anaphoricity_model)
        timer.stop("anaphoricity model")

    def clear(self):
        self.pair_vectors = self.pair_ids = None
        self.mention_vectors = self.mention_ids = None

    def get_pair_features(self, m1, m2):
        i = self.pair_ids[(m1, m2)] if (m1, m2) in self.pair_ids else \
            self.pair_ids[(m2, m1)]
        return self.pair_vectors[i]

    def get_mention_features(self, m):
        return self.mention_vectors[self.mention_ids[m]][np.newaxis]


def run_static_model(features, model):
    if model is None:
        return features

    vectors = []
    for i in range(1 + features.shape[0] / BATCH_SIZE):
        timer.start("pair model")
        vectors.append(model.predict_on_batch(
            {'X': features[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]})[0])
    vectors = np.vstack(vectors)
    assert vectors.shape[0] == features.shape[0]
    return vectors


def get_possible_pairs(probable_pairs):
    m_to_maxcluster = {}
    for m1, m2 in probable_pairs:
        c1 = m_to_maxcluster[m1] if m1 in m_to_maxcluster else {m1}
        c2 = m_to_maxcluster[m2] if m2 in m_to_maxcluster else {m2}
        if c1 != c2:
            c = c1 | c2
            for m in c:
                m_to_maxcluster[m] = c

    maxclusters = set()
    for mc in m_to_maxcluster.values():
        mc = tuple(mc)
        if mc not in maxclusters:
            maxclusters.add(mc)

    mentions = set()
    for m1, m2 in probable_pairs:
        mentions.add(m1)
        mentions.add(m2)
    assert len(mentions) == sum(len(mc) for mc in maxclusters)

    possible_pairs = []
    for mc in maxclusters:
        for m1 in mc:
            for m2 in mc:
                if m1 < m2:
                    possible_pairs.append((m1, m2))
    return possible_pairs


def write_probable_pairs(dataset_name, action_space_path, scores):
    probable_pairs = {}
    margin_removals = 0
    total_pairs = 0
    total_size = 0
    for did in utils.logged_loop(scores):
        doc_scores = scores[did]
        pairs = sorted([pair for pair in doc_scores.keys() if pair[0] != -1],
                       key=lambda pr: doc_scores[pr] - (-1 - 0.3*doc_scores[(-1, pr[1])]),
                       reverse=True)

        total_pairs += len(pairs)
        probable_pairs[did] = []
        for pair in pairs:
            score = doc_scores[pair] - (-1 - 0.3*doc_scores[(-1, pair[1])])
            if score < SCORE_THRESHOLD:
                break
            probable_pairs[did].append(pair)

        max_scores = {}
        for pair in probable_pairs[did]:
            if pair[1] not in max_scores:
                max_scores[pair[1]] = max(doc_scores[pair], -1 - 0.3*doc_scores[(-1, pair[1])])
            else:
                max_scores[pair[1]] = max(max_scores[pair[1]], doc_scores[pair])
        margin_removals += len(probable_pairs[did])
        probable_pairs[did] = [p for p in probable_pairs[did] if
                               doc_scores[p] - max_scores[p[1]] > MARGIN_THRESHOLD]
        margin_removals -= len(probable_pairs[did])
        total_size += len(probable_pairs[did])

    print("num docs:", len(scores))
    print("avg size without filter: {:.1f}".format(total_pairs / float(len(scores))))
    print("avg size: {:.1f}".format(total_size / float(len(scores))))
    print("margin removals size: {:.1f}".format(margin_removals / float(len(scores))))
    utils.write_pickle(probable_pairs, action_space_path + dataset_name + '_probable_pairs.pkl')
    shutil.copyfile('clustering_preprocessing.py',
                    action_space_path + 'clustering_preprocessing.py')


def write_action_spaces(dataset_name, action_space_path, model_path, ltr=False):
    output_file = action_space_path + dataset_name + "_action_space.pkl"
    print("Writing candidate actions to " + output_file)
    scores = utils.load_pickle(model_path + dataset_name + "_scores.pkl")
    write_probable_pairs(dataset_name, action_space_path, scores)
    probable_pairs = utils.load_pickle(action_space_path + dataset_name + '_probable_pairs.pkl')

    possible_pairs_total = 0
    action_spaces = []
    for did in scores:
        if did in probable_pairs:
            actions = defaultdict(list)
            for (m1, m2) in probable_pairs[did]:
                actions[m2].append(m1)
            if ltr:
                x = (ana1, ants1)
                y = (ana2, ants2)
                actions = sorted(actions.items(), key=functools.cmp_to_key(lambda x, y:
                                 -1 if (ana1, ana2) in scores[did] else 1))
                for i in range(len(actions) - 1):
                    assert (actions[i][0], actions[i + 1][0]) in scores[did]
            else:
                actions = sorted(actions.items(), key=lambda ana, ants:
                                 max(scores[did][(ant, ana)] - scores[did][(-1, ana)]
                                     for ant in ants))
            possible_pairs = get_possible_pairs(probable_pairs[did])
            possible_pairs_total += len(possible_pairs)
            action_spaces.append(ActionSpace(did, actions, possible_pairs))
    utils.write_pickle(action_spaces, output_file)


def main(ranking_model):
    write_action_spaces("train", directories.ACTION_SPACE,
                        directories.MODELS + ranking_model + "/")
    write_action_spaces("dev", directories.ACTION_SPACE,
                        directories.MODELS + ranking_model + "/")
    write_action_spaces("test", directories.ACTION_SPACE,
                        directories.MODELS + ranking_model + "/")

