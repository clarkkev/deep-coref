import timer
import utils
import directories
import numpy as np


MENTION_TYPES = {
    "PRONOMINAL": 0,
    "NOMINAL": 1,
    "PROPER": 2,
    "LIST": 3
}
MENTION_NUM, SENTENCE_NUM, START_INDEX, END_INDEX, MENTION_TYPE, CONTAINED = 0, 1, 2, 3, 4, 5


def make_mention_array(m):
    return np.array([m["mention_num"],
                     m["sent_num"],
                     m["start_index"],
                     m["end_index"],
                     MENTION_TYPES[m["mention_type"]],
                     m['contained-in-other-mention']], dtype='int32')


class DatasetColumn:
    def __init__(self, name, columns=None):
        self.name = name
        self.data = []
        self.active = not columns or name in columns

    def append(self, arr):
        if self.active:
            self.data.append(arr)

    def write(self, path):
        if self.active:
            self.data = np.array(self.data, dtype='bool') \
                if self.name == 'y' or self.name == 'pf' else np.vstack(self.data)
            print("Writing {:}, dtype={:}, size={:}".format(self.name, str(self.data.dtype),
                                                            utils.sizeof_fmt(self.data.nbytes)))
            np.save(path + self.name, self.data)


class DocumentDataBuilder:
    def __init__(self, columns=None):
        self.columns = columns
        self.mention_inds = DatasetColumn('dmi', columns)
        self.pair_inds = DatasetColumn('dpi', columns)
        self.features = DatasetColumn('df', columns)
        self.genres = utils.load_pickle(directories.MISC + 'genres.pkl')

    def add_doc(self, ms, me, ps, pe, features):
        self.mention_inds.append(np.array([ms, me], dtype='int32'))
        self.pair_inds.append(np.array([ps, pe], dtype='int32'))
        self.features.append(np.array(
            one_hot(np.array(self.genres[features["source"]]), len(self.genres))[0], dtype='int32'))

    def write(self, dataset_name):
        path = directories.DOC_DATA + dataset_name + '/'
        if not self.columns:
            utils.rmkdir(path)
        self.mention_inds.write(path)
        self.pair_inds.write(path)
        self.features.write(path)


class MentionDataBuilder:
    def __init__(self, columns=None):
        self.columns = columns
        self.words = DatasetColumn('mw', columns)
        self.spans = DatasetColumn('msp', columns)
        self.features = DatasetColumn('mf', columns)
        self.mention_nums = DatasetColumn('mnum', columns)
        self.mention_ids = DatasetColumn('mid', columns)
        self.dids = DatasetColumn('mdid', columns)

    def add_mention(self, m, vectors, doc_vector):
        self.features.append(make_mention_array(m))
        self.mention_nums.append(np.array(m["mention_num"], dtype='int32'))
        self.mention_ids.append(np.array(m["mention_id"], dtype='int32'))
        self.dids.append(np.array(m["doc_id"], dtype='int32'))

        s = m["sentence"]
        head_index = m["head_index"]
        start_index = m["start_index"]
        end_index = m["end_index"]

        def get_word(i): return vectors.missing if i < 0 or i >= len(s) else vectors[s[i]]
        self.words.append(np.array([
            get_word(head_index),
            get_word(start_index),
            get_word(end_index - 1),
            get_word(start_index - 1),
            get_word(end_index),
            get_word(start_index - 2),
            get_word(end_index + 1),
            vectors[m["dep_parent"]],
            vectors["dep=" + m["dep_relation"]]
        ], dtype='int32'))

        def span_vector(start, end):
            start = max(min(start, len(s) - 1), 0)
            end = max(min(end, len(s) - 1), 0)
            vs = [vectors.vectors[vectors[w]] for w in s[start:end]]
            return np.array(np.zeros(vectors.d) if len(vs) == 0 else np.mean(vs, axis=0),
                            dtype='float32')
        self.spans.append(np.array(np.concatenate([
            span_vector(start_index, end_index),
            span_vector(start_index - 5, start_index),
            span_vector(end_index, end_index + 5),
            span_vector(0, len(s)),
            doc_vector
        ])))

    def write(self, dataset_name):
        path = directories.MENTION_DATA + dataset_name + '/'
        if not self.columns:
            utils.rmkdir(path)
        self.words.write(path)
        self.spans.write(path)
        self.features.write(path)
        self.mention_nums.write(path)
        self.mention_ids.write(path)
        self.dids.write(path)

    def size(self):
        return max(len(self.words.data),
                   len(self.spans.data),
                   len(self.features.data),
                   len(self.mention_nums.data),
                   len(self.mention_ids.data),
                   len(self.dids.data))


class PairDataBuilder:
    def __init__(self, columns=None):
        self.columns = columns
        self.pair_indices = DatasetColumn('pi', columns)
        self.pair_features = DatasetColumn('pf', columns)
        self.y = DatasetColumn('y', columns)
        self.pair_ids = DatasetColumn('pmid', columns)
        self.current_did = -1
        self.current_mid2 = -1
        self.current_size = 0

    def add_pair(self, y, i1, i2, did, mid1, mid2, features):
        if self.current_did != did or self.current_mid2 != mid2:
            self.current_did = did
            self.current_mid2 = mid2
            self.current_size = 0
        self.current_size += 1

        self.y.append(y)
        self.pair_indices.append(np.array([i1, i2], dtype='int32'))
        self.pair_features.append(np.array(features, dtype='bool'))
        self.pair_ids.append(np.array([did, mid1, mid2], dtype='int32'))

    def write(self, dataset_name):
        path = directories.PAIR_DATA + dataset_name + '/'
        if not self.columns:
            utils.rmkdir(path)
        self.pair_indices.write(path)
        self.pair_features.write(path)
        self.y.write(path)
        self.pair_ids.write(path)

    def size(self):
        return max(len(self.pair_indices.data),
                   len(self.pair_features.data),
                   len(self.y.data),
                   len(self.pair_ids.data))


class Dataset:
    def __init__(self, dataset_name, model_props, word_vectors):
        self.model_props = model_props
        self.name = dataset_name
        mentions_path = directories.MENTION_DATA + dataset_name + '/'
        pair_path = directories.PAIR_DATA + dataset_name + '/'
        docs_path = directories.DOC_DATA + dataset_name + '/'

        self.words = np.load(mentions_path + 'mw.npy')
        if not model_props.use_dep_reln:
            self.words = self.words[:, :-1]
        if self.model_props.use_spans:
            self.spans = np.load(mentions_path + 'msp.npy')
            if not model_props.use_doc_embedding:
                self.spans[:, -self.spans.shape[1] / 5:] = 0
        self.mention_features = np.load(mentions_path + 'mf.npy')
        self.document_features = np.load(docs_path + 'df.npy')
        self.pair_indices = np.load(pair_path + 'pi.npy')
        self.pair_features = np.load(pair_path + 'pf.npy')[:, model_props.active_pair_features]
        self.y = np.load(pair_path + 'y.npy')

        self.doc_sizes = {}
        doc_pairs = np.load(docs_path + 'dpi.npy')
        doc_mentions = np.load(docs_path + 'dmi.npy')
        for did in np.arange(doc_pairs.shape[0]):
            ms, me = doc_mentions[did]
            self.doc_sizes[did] = me - ms

        mids = np.load(mentions_path + 'mid.npy')
        dids = np.load(mentions_path + 'mdid.npy')
        pids = np.hstack([dids[self.pair_indices[:, 0]],
                          mids[self.pair_indices[:, 0]],
                          mids[self.pair_indices[:, 1]]])
        self.pair_ids_to_index = {tuple(pids[i]): i for i in range(pids.shape[0])}
        self.mention_ids_to_index = {(dids[i, 0], mids[i, 0]): i for i in range(mids.size)}

        self.word_vectors = np.asarray(word_vectors)
        self.vector_size = self.word_vectors.shape[1]

    def vectorize_mentions(self, did, ms):
        res = self.featurize_mention([self.mention_ids_to_index[(did, m)] for m in ms], did), \
               {m: i for i, m in enumerate(ms)}
        return res

    def vectorize_pairs(self, did, pairs):
        pair_ids = {}
        batch = []
        for i, (m1, m2) in enumerate(pairs):
            if (did, m1, m2) not in self.pair_ids_to_index:
                m1, m2 = m2, m1
            pair_ids[(m1, m2)] = pair_ids[(m2, m1)] = i
            batch.append(self.pair_ids_to_index[(did, m1, m2)])
        pair_indices = self.pair_indices[batch]
        m1 = pair_indices[:, 0]
        m2 = pair_indices[:, 1]
        return self.featurize_pairs(m1, m2, batch, did), pair_ids

    def get_vectors(self, words):
        return np.vstack([np.reshape(self.word_vectors[words[i]],
                                     (words.shape[1] * self.vector_size))
                          for i in range(words.shape[0])])

    def featurize_mention(self, m, did):
        return np.hstack([
            self.spans[m],
            self.get_vectors(self.words[m]),
            get_dense_features_anaphoricity(
                self.mention_features[m],
                self.document_features[did],
                self.doc_sizes[did],
                self.model_props)])

    def featurize_pairs(self, m1, m2, batch, did):
        return np.hstack([
                self.spans[m1],
                self.get_vectors(self.words[m1]),
                self.spans[m2],
                self.get_vectors(self.words[m2]),
                get_dense_features(self.mention_features[m1],
                                   self.mention_features[m2],
                                   self.pair_features[batch],
                                   self.document_features[did],
                                   self.doc_sizes[did],
                                   self.model_props)])


class DocumentBatchedDataset:
    """
    Shuffling and then iterating through all mention pairs in the dataset has two problems:
        1. For the sake of efficiency we want to compute a representation for a mention (in our
           case by looking up some word embeddings and applying a hidden layer) once for every
           mention instead of once for every pair of mentions.
        2. For mention-ranking models, all pairs involving the current candidate anaphor must be
           in the same batch.
    We deal with this by instead using each document as a batch, except for large documents, which
    we split into chunks.
    """
    def __init__(self, dataset_name, model_props, max_pairs=10000, with_ids=False):
        self.name = dataset_name
        self.model_props = model_props
        self.with_ids = with_ids
        self.anaphoricity = model_props.anaphoricity
        self.anaphoricity_only = model_props.anaphoricity_only

        mentions_path = directories.MENTION_DATA + dataset_name + '/'
        pair_path = directories.PAIR_DATA + dataset_name + '/'
        docs_path = directories.DOC_DATA + dataset_name + '/'

        self.words = np.load(mentions_path + 'mw.npy')
        if not self.model_props.use_dep_reln:
            self.words = self.words[:, :-1]
        if self.model_props.use_spans:
            self.spans = np.load(mentions_path + 'msp.npy')
            if not model_props.use_doc_embedding:
                print(-self.spans.shape[1] / 5)
                self.spans[:, -self.spans.shape[1] / 5:] = 0

        self.mention_features = np.load(mentions_path + 'mf.npy')
        self.document_features = np.load(docs_path + 'df.npy')
        if not model_props.use_genre:
            self.document_features = np.zeros((self.document_features.shape[0], 1))

        self.pair_features = np.load(pair_path + 'pf.npy')[:, model_props.active_pair_features]
        self.y = np.load(pair_path + 'y.npy')
        if with_ids:
            self.pair_ids = np.load(pair_path + 'pmid.npy')
            self.mention_ids = np.load(mentions_path + 'mid.npy')

        doc_pairs = np.load(docs_path + 'dpi.npy')
        doc_mentions = np.load(docs_path + 'dmi.npy')

        self.pair_nums = []
        for did in np.arange(doc_pairs.shape[0]):
            ms, me = doc_mentions[did]
            if me != ms:
                pair_antecedents = np.concatenate([np.arange(ana)
                                                   for ana in range(0, me - ms)])
                pair_anaphors = np.concatenate([ana *
                                                np.ones(ana, dtype='int32')
                                                for ana in range(0, me - ms)])
                self.pair_nums += [np.array(p) for p in zip(pair_antecedents, pair_anaphors)]
        self.pair_nums = np.vstack(self.pair_nums)

        self.doc_sizes = {}
        self.n_pairs = 0
        self.n_anaphors = 0
        self.n_anaphoric_anaphors = 0
        self.batches = []

        timer.start("preprocess_dataset")
        for did in np.arange(doc_pairs.shape[0]):
            ps, pe = doc_pairs[did]
            ms, me = doc_mentions[did]
            min_anaphor = 1
            min_pair = 0
            self.n_anaphors += me - ms
            self.doc_sizes[did] = me - ms

            while min_anaphor < me - ms:
                max_anaphor = min(new_max_anaphor(min_anaphor, max_pairs), me - ms)
                max_pair = min(max_anaphor * (max_anaphor - 1) / 2, pe - ps)

                mentions = np.arange(ms, ms + max_anaphor)
                antecedents = np.arange(max_anaphor - 1)
                anaphors = np.arange(min_anaphor, max_anaphor)
                pairs = np.arange(ps + min_pair, ps + max_pair, dtype=int)
                pair_antecedents = np.concatenate([np.arange(ana)
                                                   for ana in range(min_anaphor, max_anaphor)])
                pair_anaphors = np.concatenate([(ana - min_anaphor) *
                                                np.ones(ana, dtype='int32')
                                                for ana in range(min_anaphor, max_anaphor)])

                positive, negative = [], []
                ana_to_pos, ana_to_neg = {}, {}

                assert pair_anaphors.size == self.y[pairs].size
                ys = self.y[pairs]
                for i, (ana, y) in enumerate(zip(pair_anaphors, ys)):
                    labels = positive if y == 1 else negative
                    ana_to_ind = ana_to_pos if y == 1 else ana_to_neg
                    if ana not in ana_to_ind:
                        ana_to_ind[ana] = [len(labels), len(labels)]
                    else:
                        ana_to_ind[ana][1] = len(labels)
                    labels.append(i)

                pos_starts, pos_ends, neg_starts, neg_ends = [], [], [], []
                anaphoricities = []
                for ana in range(0, max_anaphor - min_anaphor):
                    if ana in ana_to_pos:
                        start, end = ana_to_pos[ana]
                        pos_starts.append(start)
                        pos_ends.append(end + 1)
                        anaphoricities.append(1)
                    else:
                        anaphoricities.append(0)
                    if ana in ana_to_neg:
                        start, end = ana_to_neg[ana]
                        neg_starts.append(start)
                        neg_ends.append(end + 1)

                starts, ends, costs = [], [], [],
                reindex = []
                pair_pos, anaphor_pos = 0, len(pairs)
                i, j = 0, 0
                for ana in range(0, max_anaphor - min_anaphor):
                    ana_labels = []
                    ana_reindex = []
                    start = i
                    for ant in range(0, ana + min_anaphor):
                        ana_labels.append(ys[j])
                        i += 1
                        j += 1
                        ana_reindex.append(pair_pos)
                        pair_pos += 1
                    if model_props.anaphoricity:
                        i += 1
                        ana_reindex.append(anaphor_pos)
                        anaphor_pos += 1
                    end = i
                    ana_labels = np.array(ana_labels)
                    anaphoric = ana_labels.sum() > 0
                    if (model_props.anaphoricity or anaphoric) and end > start + 1:
                        starts.append(start)
                        ends.append(end)
                        reindex += ana_reindex
                        self.n_anaphoric_anaphors += 1
                    else:
                        i = start
                        continue

                    assert anaphoric == anaphoricities[ana]
                    if model_props.anaphoricity:
                        if anaphoric:
                            ana_costs = np.append(model_props.WL * (ana_labels ^ 1), model_props.FN)
                        else:
                            ana_costs = np.append(model_props.FL * np.ones_like(ana_labels), 0)
                    else:
                        ana_costs = ana_labels ^ 1
                    costs += list(ana_costs)
                reindex = np.array(reindex, dtype='int32')

                self.batches.append((did, mentions, antecedents, anaphors,
                                     pairs, pair_antecedents, pair_anaphors,
                                     np.array(positive, dtype='int32'),
                                     np.array(negative, dtype='int32'),
                                     np.array(pos_starts, dtype='int32'),
                                     np.array(pos_ends, dtype='int32'),
                                     np.array(neg_starts, dtype='int32'),
                                     np.array(neg_ends, dtype='int32'),
                                     np.array(anaphoricities, dtype='int32'),
                                     reindex,
                                     np.array(starts, dtype='int32'),
                                     np.array(ends, dtype='int32'),
                                     np.array(costs, dtype='float32')))
                self.n_pairs += len(pairs)

                min_anaphor = max_anaphor
                min_pair = max_pair
        timer.stop("preprocess_dataset")

        self.n_batches = len(self.batches)
        self.pairs_per_batch = float(self.n_pairs) / self.n_batches
        self.anaphoric_anaphors_per_batch = float(self.n_anaphoric_anaphors) / self.n_batches
        self.anaphors_per_batch = float(self.n_anaphors) / self.n_batches

        if model_props.ranking:
            self.scale_factor = self.anaphors_per_batch if model_props.anaphoricity else \
                self.anaphoric_anaphors_per_batch
        elif model_props.top_pairs:
            self.scale_factor = 10 * self.anaphors_per_batch
        else:
            self.scale_factor = self.pairs_per_batch
        self.anaphoricity_scale_factor = 20 * self.anaphors_per_batch \
            if self.model_props.top_pairs else 50 * self.anaphors_per_batch

    def shuffle(self):
        np.random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            timer.start("minibatch_prep")
            did, mentions, antecedents, anaphors,\
                pairs, pair_antecedents, pair_anaphors,\
                positive, negative,\
                pos_starts, pos_ends, neg_starts, neg_ends,\
                anaphoricities,\
                reindex, starts, ends, costs = batch
            document_features = self.document_features[did]

            X = {}
            X['words'] = self.words[mentions]
            if self.model_props.use_spans:
                X['spans'] = self.spans[mentions]

            X['anaphors'] = anaphors[:, np.newaxis]
            if not self.anaphoricity_only:
                X['antecedents'] = antecedents[:, np.newaxis]
                X['pair_antecedents'] = pair_antecedents[:, np.newaxis]
                X['pair_anaphors'] = pair_anaphors[:, np.newaxis]
                X['pair_features'] = get_dense_features(
                    self.mention_features[pair_antecedents + mentions[0]] + antecedents[0],
                    self.mention_features[pair_anaphors + mentions[0] + anaphors[0]],
                    self.pair_features[pairs], document_features, self.doc_sizes[did],
                    self.model_props)

            if self.model_props.top_pairs:
                X['score_inds'] = np.concatenate([positive, negative])[:, np.newaxis]
                X['starts'] = np.concatenate([pos_starts, positive.size + neg_starts])[:, np.newaxis]
                X['ends'] = np.concatenate([pos_ends, positive.size + neg_ends])[:, np.newaxis]
                X['y'] = np.concatenate([np.ones(pos_starts.size),
                                         np.zeros(neg_starts.size)])[:, np.newaxis]
            elif self.model_props.ranking:
                X['reindex'] = reindex[:, np.newaxis]
                X['starts'] = starts[:, np.newaxis]
                X['ends'] = ends[:, np.newaxis]
                X['costs'] = costs[:, np.newaxis]
                X['y'] = np.zeros((starts.size, 1))
                if self.model_props.use_rewards:
                    X['cost_ptrs'] = costs
            else:
                X['y'] = self.y[pairs][:, np.newaxis]

            if self.with_ids:
                anaphor_ids = self.mention_ids[mentions][anaphors]
                if self.model_props.ranking:
                    antecedent_ids = self.mention_ids[mentions][antecedents]
                    pair_antecedent_ids = antecedent_ids[pair_antecedents]
                    pair_anaphor_ids = anaphor_ids[pair_anaphors]
                    pair_ids = np.hstack([pair_antecedent_ids, pair_anaphor_ids])
                    anaphor_ids = np.hstack([-1 * np.ones_like(anaphor_ids), anaphor_ids])
                    all_ids = np.vstack([pair_ids, anaphor_ids])
                    X['ids'] = all_ids[reindex]
                    X['did'] = self.pair_ids[pairs][0, 0]
                else:
                    X['ids'] = self.pair_ids[pairs]
                    X['anaphor_ids'] = anaphor_ids

            if self.anaphoricity:
                X['anaphoricities'] = anaphoricities[:, np.newaxis]
                X['mention_features'] = get_dense_features_anaphoricity(
                    self.mention_features[mentions], document_features, self.doc_sizes[did],
                    self.model_props)

            timer.stop("minibatch_prep")
            yield X


def new_max_anaphor(n, k):
    # find m such that sum from i=n to m-1 is < k
    # i.e., total number of pairs with anaphor num between n and m (exclusive) < k
    return max(1, int(np.floor(0.5 * (1 + np.sqrt(8 * k + 4 * n * n - 4 * n + 1)))))


def get_dense_features_anaphoricity(mention_features, document_features, doc_size, model_props):
    fs = np.hstack(get_mention_features(mention_features, doc_size, model_props))
    tiled = np.tile(document_features[np.newaxis, :], (fs.shape[0], 1))
    return np.concatenate((fs, tiled), axis=1)


def get_dense_features(m1_features, m2_features, pair_features, document_features, doc_size,
                       model_props):
    fs = np.hstack([pair_features] +
                   (get_distance_features(m1_features, m2_features)
                    if model_props.use_distance else []) +
                   get_mention_features(m1_features, doc_size, model_props) +
                   get_mention_features(m2_features, doc_size, model_props))
    tiled = np.tile(document_features[np.newaxis, :], (fs.shape[0], 1))
    return np.concatenate((fs, tiled), axis=1)


def get_mention_features(m, doc_size, model_props):
    features = []
    if model_props.use_mention_type:
        features.append(one_hot(m[:, MENTION_TYPE], 4))
    if model_props.use_length:
        features.append(distance(np.subtract(m[:, END_INDEX] - m[:, START_INDEX], 1)))
    if model_props.use_position:
        features.append(m[:, MENTION_NUM][:, np.newaxis] / float(doc_size))
    if model_props.use_distance:
        features.append(m[:, CONTAINED][:, np.newaxis])
    return features


def get_distance_features(m1, m2):
    return [distance(m2[:, SENTENCE_NUM] - m1[:, SENTENCE_NUM]),
            distance(np.subtract(m2[:, MENTION_NUM] - m1[:, MENTION_NUM], 1)),
            ((m2[:, SENTENCE_NUM] == m1[:, SENTENCE_NUM]) &
            (m1[:, END_INDEX] > m2[:, START_INDEX]))[:, np.newaxis]]


def one_hot(a, n):
    oh = np.zeros((a.size, n))
    oh[np.arange(a.size), a] = 1
    return oh


def distance(a):
    d = np.zeros((a.size, 11))
    d[a == 0, 0] = 1
    d[a == 1, 1] = 1
    d[a == 2, 2] = 1
    d[a == 3, 3] = 1
    d[a == 4, 4] = 1
    d[(5 <= a) & (a < 8), 5] = 1
    d[(8 <= a) & (a < 16), 6] = 1
    d[(16 <= a) & (a < 32), 7] = 1
    d[(a >= 32) & (a < 64), 8] = 1
    d[a >= 64, 9] = 1
    d[:, 10] = np.clip(a, 0, 64) / 64.0
    return d
