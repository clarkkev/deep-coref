import utils

DATA = './data/'

RAW = DATA + 'data_raw/'
MODELS = DATA + 'models/'
CLUSTERERS = DATA + 'clusterers/'
DOCUMENTS = DATA + 'documents/'
ACTION_SPACES_BASE = DATA + 'action_spaces/'
GOLD = DATA + 'gold/'
MISC = DATA + 'misc/'

CHINESE = False
PRETRAINED_WORD_VECTORS = DATA + 'polyglot_64d.txt' if CHINESE else DATA + 'w2v_50d.txt'

FEATURES_BASE = DATA + 'features/'
MENTION_DATA = FEATURES_BASE + 'mention_data/'
RELEVANT_VECTORS = MENTION_DATA
PAIR_DATA = FEATURES_BASE + 'mention_pair_data/'
DOC_DATA = FEATURES_BASE + 'doc_data/'

ACTION_SPACE_NAME = 'action_spaces/'
ACTION_SPACE = ACTION_SPACES_BASE + ACTION_SPACE_NAME

assert DATA[-1] == '/'
assert ACTION_SPACE_NAME[-1] == '/'

utils.mkdir(MISC)
utils.mkdir(FEATURES_BASE)
utils.mkdir(MENTION_DATA)
utils.mkdir(PAIR_DATA)
utils.mkdir(DOC_DATA)
utils.mkdir(MODELS)
utils.mkdir(CLUSTERERS)
utils.mkdir(DOCUMENTS)
utils.mkdir(ACTION_SPACES_BASE)
utils.mkdir(ACTION_SPACE)

