import util
import directories
import os
from keras.regularizers import l2
from keras.optimizers import RMSprop


class MentionRankingProps:
    def __init__(self,
                 mode='ranking',
                 # model initialization
                 load_weights_from=None, weights_file=None,
                 # neural network architecture
                 layer_sizes=None, activation='relu', dropout=0.5, freeze_embeddings=False,
                 # error penalties for ranking objective
                 FN=0.8, FL=0.4, WL=1.0,
                 # learning rates
                 classification_lr=0.002, top_pairs_lr=0.0001, ranking_lr=0.000005,
                 # which speaker and string-matching features
                 pair_features=None,
                 # mention features
                 use_length=True, use_mention_type=True,  use_position=True, use_dep_reln=False,
                 # distance and genre features
                 use_distance=True, use_genre=True,
                 # averaged word embedding features
                 use_spans=True, use_doc_embedding=True):
        if layer_sizes is None:
            layer_sizes = [1000, 500, 500]
        if pair_features is None:
            pair_features=[
               # speaker features
               "same-speaker",
               "antecedent-is-mention-speaker",
               "mention-is-antecedent-speaker",
               # string-matching features
               "relaxed-head-match",
               "exact-string-match",
               "relaxed-string-match",
           ]

        self.load_weights_from = load_weights_from
        self.weights_file = weights_file
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.dropout = dropout
        self.freeze_embeddings = freeze_embeddings
        self.FN, self.FL, self.WL = FN, FL, WL
        self.ranking_lr = ranking_lr
        self.top_pairs_lr = top_pairs_lr
        self.classification_lr = classification_lr

        self.use_length = use_length
        self.use_mention_type = use_mention_type
        self.use_position = use_position
        self.use_dep_reln = use_dep_reln
        self.use_distance = use_distance
        self.use_genre = use_genre
        self.use_spans = use_spans
        self.use_doc_embedding = use_doc_embedding

        if os.path.exists(directories.MISC + 'pair_feature_names.pkl'):
            name_mapping = util.load_pickle(directories.MISC + 'pair_feature_names.pkl')
            self.active_pair_features = sorted([name_mapping[f] for f in pair_features])

        self.set_mode(mode)

    def set_mode(self, mode):
        self.mode = mode
        self.top_pairs = False
        self.ranking = True
        self.anaphoricity = True
        self.anaphoricity_only = False
        self.regularization = 1e-5 if self.top_pairs or self.ranking else 1e-6

        if mode == 'ranking':
            pass
        elif mode == 'ranking_noana':
            self.anaphoricity = False
        elif mode == 'classification':
            self.ranking = False
        elif mode == 'top_pairs':
            self.ranking = False
            self.top_pairs = True
        elif mode == 'pairwise':
            self.ranking = False
            self.anaphoricity = False
        elif mode == 'anaphoricity':
            self.ranking = False
            self.anaphoricity_only = True
        else:
            raise ValueError("Unkown mode " + mode)

    def get_regularizer(self):
        return None if self.regularization is None else l2(self.regularization)

    def get_optimizer(self):
        if self.ranking:
            return RMSprop(self.ranking_lr, epsilon=1e-5)
        return RMSprop(lr=self.top_pairs_lr if self.top_pairs else self.classification_lr,
                       epsilon=1e-5)

    def write(self, path):
        util.write_pickle(self.__dict__, path)


class ClusterRankingProps:
    def __init__(self,

                 top_layers=3,
                 learnable_layers=3,
                 risk_objective=True,
                 randomize_weights=False,
                 pooling='maxavg',
                 input_dropout=0,
                 dropout=0.2,
                 learning_rate=1e-7):
        assert pooling in ['max', 'avg', 'maxavg']
        self.top_layers = top_layers
        self.learnable_layers = learnable_layers
        self.risk_objective = risk_objective
        self.randomize_weights = randomize_weights
        self.pooling = pooling
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.learning_rate = learning_rate

        self.single_size = 855 if directories.CHINESE else 674
        self.pair_size = 1733 if directories.CHINESE else 1370
        self.static_layers = top_layers - learnable_layers
        if self.static_layers == 0:
            self.anaphoricity_input_size = self.single_size
            self.pair_input_size = self.pair_size
        elif self.static_layers == 1:
            self.anaphoricity_input_size = self.pair_input_size = 1000
        else:
            self.anaphoricity_input_size = self.pair_input_size = 500
