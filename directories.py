import util

DATA = './data/'

RAW = DATA + 'raw/'
MODELS_BASE = DATA + 'models/'
CLUSTERER_BASE = DATA + 'clusterers/'
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

MODEL_NAME = 'model/'
MODEL = MODELS_BASE + MODEL_NAME

CLUSTERER_NAME = 'clusterer/'
CLUSTERER = CLUSTERER_BASE + CLUSTERER_NAME

ACTION_SPACE_NAME = 'action_spaces/'
ACTION_SPACE = ACTION_SPACES_BASE + ACTION_SPACE_NAME

assert DATA[-1] == '/'
assert ACTION_SPACE_NAME[-1] == '/'
assert MODEL_NAME[-1] == '/'
assert CLUSTERER[-1] == '/'

util.mkdir(MISC)
util.mkdir(FEATURES_BASE)
util.mkdir(MENTION_DATA)
util.mkdir(PAIR_DATA)
util.mkdir(DOC_DATA)
util.mkdir(MODELS_BASE)
util.mkdir(CLUSTERER_BASE)
util.mkdir(MODEL)
util.mkdir(CLUSTERER)
util.mkdir(DOCUMENTS)
util.mkdir(ACTION_SPACES_BASE)
util.mkdir(ACTION_SPACE)


def set_model_name(model_name):
    global MODEL_NAME, MODEL
    MODEL_NAME = model_name + '/'
    MODEL = MODELS_BASE + MODEL_NAME
    util.mkdir(MODEL)
