import directories
import build_dataset
import preprocessing
import clustering_preprocessing
import clustering_learning
import pairwise_learning
import model_properties
import timer
import document
from document import Document


def setup():
    preprocessing.main()
    build_dataset.build_datasets(reduced=True)
    build_dataset.build_datasets(reduced=False)
    document.main()


def main(model_props=None, cluster_props=None):
    if model_props is None:
        model_props = model_properties.MentionRankingProps()
    if cluster_props is None:
        cluster_props = model_properties.ClusterRankingProps()

    directories.set_model_name('classification')
    model_props.load_weights_from = None
    model_props.set_mode('classification')
    pairwise_learning.main(model_props=model_props, n_epochs=150)

    directories.set_model_name('top_pairs')
    model_props.set_mode('top_pairs')
    model_props.load_weights_from = 'classification'
    model_props.weights_file = 'weights_140'
    timer.clear()
    pairwise_learning.main(model_props=model_props, n_epochs=50)

    directories.set_model_name('ranking')
    model_props.set_mode('ranking')
    model_props.load_weights_from = 'top_pairs'
    model_props.weights_file = 'weights_40'
    timer.clear()
    pairwise_learning.main(model_props=model_props, n_epochs=50)

    model_props.load_weights_from = 'ranking'
    pairwise_learning.main(model_props=model_props, write_scores=True, test_only=True)
    pairwise_learning.main(model_props=model_props, write_scores=True, test_only=True,
                           validate="test")

    clustering_preprocessing.main('ranking')
    cluster_props.load_weights_from = 'ranking'
    cluster_props.weights_file = 'weights_40'
    timer.celar()
    clustering_learning.main(cluster_props)

if __name__ == '__main__':
    setup()
    main()
