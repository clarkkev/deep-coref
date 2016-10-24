import utils
import directories
import os
from keras.regularizers import l2
from keras.optimizers import RMSprop


class MentionRankingProps:
    def __init__(self, name=None, mode='ranking',
                 # model initialization
                 load_weights_from=None, weights_file=None,
                 # neural network architecture
                 layer_sizes=None, activation='relu', dropout=0.5, freeze_embeddings=False,
                 # error penalties for heuristic ranking objective
                 FN=0.8, FL=0.5 if directories.CHINESE else 0.4, WL=1.0,
                 # learning rates
                 all_pairs_lr=0.002, top_pairs_lr=0.0002, ranking_lr=0.000002,
                 reinforce_lr=0.000002, reward_rescaling_lr=0.00002,
                 # which speaker and string-matching features to use
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
        self.reinforce_lr = reinforce_lr
        self.reward_rescaling_lr = reward_rescaling_lr
        self.top_pairs_lr = top_pairs_lr
        self.all_pairs_lr = all_pairs_lr

        self.use_length = use_length
        self.use_mention_type = use_mention_type
        self.use_position = use_position
        self.use_dep_reln = use_dep_reln
        self.use_distance = use_distance
        self.use_genre = use_genre
        self.use_spans = use_spans
        self.use_doc_embedding = use_doc_embedding

        if os.path.exists(directories.MISC + 'pair_feature_names.pkl'):
            name_mapping = utils.load_pickle(directories.MISC + 'pair_feature_names.pkl')
            self.active_pair_features = sorted([name_mapping[f] for f in pair_features])

        self.set_name(name)
        self.set_mode(mode)

    def set_mode(self, mode):
        self.mode = mode
        self.top_pairs = False
        self.ranking = True
        self.anaphoricity = True
        self.anaphoricity_only = False
        self.reinforce = False
        self.use_rewards = False

        if mode == 'ranking':             # mention-ranking model with heuristic loss
            pass
        elif mode == 'reward_rescaling':  # mention-ranking model with reward-rescaling loss
            self.use_rewards = True
        elif mode == 'reinforce':         # mention-ranking model with the REINFORCE algorithm
            self.use_rewards = True
            self.reinforce = True
        elif mode == 'ranking_noana':     # mention-ranking only over anaphoric mentions
            self.anaphoricity = False
        elif mode == 'all_pairs':         # all-pairs classification objective
            self.ranking = False
        elif mode == 'top_pairs':         # top-pairs classification objective
            self.ranking = False
            self.top_pairs = True
        elif mode == 'pairwise':          # binary classification over mention pairs
            self.ranking = False
            self.anaphoricity = False
        elif mode == 'anaphoricity':      # anaphoricity classification
            self.ranking = False
            self.anaphoricity_only = True
        else:
            raise ValueError("Unkown mode " + mode)

        if mode == 'ranking':
            self.lr = self.ranking_lr
        elif mode == 'reward_rescaling':
            self.lr = self.reward_rescaling_lr
        elif mode == 'reinforce':
            self.lr = self.reinforce_lr
        elif mode == 'top_pairs':
            self.lr = self.top_pairs_lr
        else:
            self.lr = self.all_pairs_lr

        self.regularization = 1e-5 if self.top_pairs or self.ranking else 1e-6

    def set_name(self, name):
        if name is not None:
            self.name = name
            self.path = directories.MODELS + name + '/'
            utils.mkdir(self.path)

    def get_regularizer(self):
        return None if self.regularization is None else l2(self.regularization)

    def get_optimizer(self):
        return RMSprop(lr=self.lr, epsilon=1e-5)

    def write(self, path):
        utils.write_pickle(self.__dict__, path)


class ClusterRankingProps:
    def __init__(self, name='clusterer',
                 # model initialization
                 load_weights_from=None, weights_file=None, randomize_weights=False,
                 # network architecture
                 top_layers=3, learnable_layers=3, pooling='maxavg', risk_objective=True,
                 # dropout and learning rates
                 input_dropout=0, dropout=0.0, learning_rate=1e-7):
        assert pooling in ['max', 'avg', 'maxavg']

        self.name = name
        self.path = directories.CLUSTERERS + '/'
        utils.mkdir(self.path)

        self.load_weights_from = load_weights_from
        self.weights_file = weights_file
        self.randomize_weights = randomize_weights
        self.top_layers = top_layers
        self.learnable_layers = learnable_layers
        self.pooling = pooling
        self.risk_objective = risk_objective
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
