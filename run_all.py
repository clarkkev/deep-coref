import directories
import build_datasets
import preprocessing
import clustering_preprocessing
import clustering_learning
import pairwise_learning
import model_properties
import document
from document import Document
import os


OVERWRITE_EXISTING_MODELS = False


def setup():
    preprocessing.main()
    build_datasets.build_datasets(reduced=True)
    build_datasets.build_datasets(reduced=False)
    document.main()


def already_trained(name, weights):
    return os.path.exists(directories.MODELS + name + '/' + weights + '.hdf5') and \
           not OVERWRITE_EXISTING_MODELS


def pretrain(model_props):
    if not already_trained('all_pairs', 'weights_140'):
        model_props.set_name('all_pairs')
        model_props.set_mode('all_pairs')
        model_props.load_weights_from = None
        pairwise_learning.train(model_props, n_epochs=150)

    if not already_trained('top_pairs', 'weights_40'):
        model_props.set_name('top_pairs')
        model_props.set_mode('top_pairs')
        model_props.load_weights_from = 'all_pairs'
        model_props.weights_file = 'weights_140'
        pairwise_learning.train(model_props, n_epochs=50)


def make_predictions(model_props, load_weights_from, datasets, save_scores=False):
    model_props.load_weights_from = load_weights_from
    model_props.weights_file = 'final_weights'
    for dataset_name in datasets:
        pairwise_learning.test(model_props=model_props, save_scores=save_scores, save_output=True,
                               dataset_name=dataset_name)


def train_clustering(cluster_props):
    clustering_preprocessing.main('ranking')
    cluster_props.load_weights_from = 'ranking'
    cluster_props.weights_file = 'best_weights'
    clustering_learning.main(cluster_props)


def train_pairwise(model_props, mode='ranking'):
    pretrain(model_props)
    model_props.set_name(mode)
    model_props.set_mode(mode)
    model_props.load_weights_from = 'top_pairs'
    model_props.weights_file = 'weights_40'
    pairwise_learning.train(model_props, n_epochs=100)


def train_and_test_pairwise(model_props, mode='ranking'):
    train_pairwise(model_props, mode=mode)
    model_props.set_name(mode)
    make_predictions(model_props, mode, ["dev", "test"])


def acl2016():
    model_props = model_properties.MentionRankingProps()
    train_pairwise(model_props)
    make_predictions(model_props, 'ranking', ["train", "dev", "test"], save_scores=True)
    train_clustering(model_properties.ClusterRankingProps())


def emnlp2016():
    model_props = model_properties.MentionRankingProps()
    train_and_test_pairwise(model_props, mode='ranking')
    train_and_test_pairwise(model_props, mode='reinforce')
    train_and_test_pairwise(model_props, mode='reward_rescaling')


def train_best_model():
    train_and_test_pairwise(model_properties.MentionRankingProps(), mode='reward_rescaling')


if __name__ == '__main__':
    setup()
    train_best_model()
