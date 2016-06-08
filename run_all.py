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


def main(model_props=None):
    if model_props is None:
        model_props = model_properties.MentionRankingProps()

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

        pairwise_learning.load_weights_from = 'ranking'
        timer.clear()
        pairwise_learning.main(model_props=model_props, n_epochs=50, write_scores=True)

        clustering_preprocessing.main('ranking')
        clustering_learning.main()

if __name__ == '__main__':
    setup()
    main()
