import directories
import timer
from custom_neural_implementations import Identity, get_summed_cross_entropy, get_sum

import numpy as np
import h5py

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding


def add_mention_reprs(graph, name, model_props):
    spans = name + '_spans'
    spans_reprs = name + '_spans_reprs'
    pair_spans_reprs = 'pair_' + name + '_spans_reprs'
    flattened_embeddings = name + '_embeddings_flattened'
    pair_embeddings_dropped = 'pair_' + name + '_embeddings_dropped'
    pair_embeddings_reprs = 'pair_' + name + '_embeddings_reprs'
    reprs = 'pair_' + name + '_reprs'
    mentions = name + 's'
    pairs = 'pair_' + name + 's'

    if model_props.use_spans:
        # get relevant spans
        graph.add_node(Identity(), name=spans, inputs=['spans', mentions], merge_mode='index')
        # put through a fully connected layer
        graph.add_node(Dense(model_props.layer_sizes[0], W_regularizer=model_props.get_regularizer()),
                       name=spans_reprs, input=spans)
        # duplicate to make pairs
        graph.add_node(Identity(), name=pair_spans_reprs, inputs=[spans_reprs, pairs],
                       merge_mode='index')

    # get relevant words
    graph.add_node(Identity(), name=flattened_embeddings,
                   inputs=['flattened_embeddings', mentions], merge_mode='index')
    # duplicate and dropout
    graph.add_node(Dropout(model_props.dropout), name=pair_embeddings_dropped,
                   inputs=[flattened_embeddings, pairs], merge_mode='index')
    # put through fully connected layer (could do this before if no word dropout)
    graph.add_node(Dense(model_props.layer_sizes[0], W_regularizer=model_props.get_regularizer()),
                   name=pair_embeddings_reprs if model_props.use_spans else reprs,
                   input=pair_embeddings_dropped)

    # combine word and span representations to get mention representation
    if model_props.use_spans:
        graph.add_node(Identity(), name=reprs,
                       inputs=[pair_spans_reprs, pair_embeddings_reprs], merge_mode='sum')


def add_anaphoricity_reprs(graph, model_props):
    spans = 'anaphoricity_spans'
    spans_reprs = 'anaphoricity_spans_reprs'
    flattened_embeddings = 'anaphoricity_embeddings_flattened'
    embeddings_dropped = 'anaphoricity_embeddings_dropped'
    embeddings_reprs = 'anaphoricity_embeddings_reprs'
    reprs = 'anaphoricity_reprs'
    mentions = 'anaphors'
    anaphoricity_features = 'anaphoricity_reprs_features'
    anaphoricity_features_reprs = 'anaphoricity_reprs_features_reprs'

    if model_props.use_spans:
        # get relevant spans
        graph.add_node(Identity(), name=spans, inputs=['spans', mentions], merge_mode='index')
        # put through a fully connected layer
        graph.add_node(Dense(model_props.layer_sizes[0], W_regularizer=model_props.get_regularizer()),
                       name=spans_reprs, input=spans)

    # get relevant words
    graph.add_node(Identity(), name=flattened_embeddings,
                   inputs=['flattened_embeddings', mentions], merge_mode='index')
    # dropout and put through fully connected layer
    graph.add_node(Dropout(model_props.dropout),
                   name=embeddings_dropped, input=flattened_embeddings)
    graph.add_node(Dense(model_props.layer_sizes[0], W_regularizer=model_props.get_regularizer()),
                   name=embeddings_reprs, input=embeddings_dropped)

    # get additional mention features
    graph.add_node(Identity(), name=anaphoricity_features, inputs=['mention_features', mentions],
                   merge_mode='index')
    # put through fully connected layer
    graph.add_node(Dense(model_props.layer_sizes[0], W_regularizer=model_props.get_regularizer()),
                   name=anaphoricity_features_reprs, input=anaphoricity_features)
    # combine span, word, and additional features
    graph.add_node(Identity(), name=reprs, inputs=[spans_reprs, embeddings_reprs,
                                                   anaphoricity_features_reprs], merge_mode='sum')


def get_top_layers(model_props, representation=False):
    top_layers = Sequential()
    top_layers.add(Activation(model_props.activation, input_shape=(model_props.layer_sizes[0],)))

    for i in range(len(model_props.layer_sizes) - 1):
        top_layers.add(Dropout(model_props.dropout))
        top_layers.add(Dense(model_props.layer_sizes[i + 1], activation=model_props.activation,
                             W_regularizer=model_props.get_regularizer()))

    if not representation:
        top_layers.add(Dropout(model_props.dropout))
        top_layers.add(Dense(1, W_regularizer=model_props.get_regularizer()))
        top_layers.add(Dense(1, W_regularizer=model_props.get_regularizer(),
                             weights=[np.array([[1]]), np.array([0])]))
        if not model_props.ranking:
            top_layers.add(Activation('sigmoid'))

    return top_layers


def get_embedding_layer(model_props, input_sizes, vectors):
    return Embedding(vectors.shape[0], vectors.shape[1], weights=[vectors],
                     input_length=input_sizes['words'],
                     trainable=not model_props.freeze_embeddings)


def build_graph(train, vectors, model_props, representation=False):
    input_sizes = {k: v.shape[1] for k, v in next(X for X in train).items() if v.ndim == 2}

    graph = Graph()
    graph.add_input(name='anaphors', input_shape=(1,), dtype='int32')

    graph.add_input(name='words', input_shape=(input_sizes['words'],), dtype='int32')
    graph.add_node(get_embedding_layer(model_props, input_sizes, vectors),
                   name='word_embeddings', input='words')
    graph.add_node(Flatten(), name='flattened_embeddings', input='word_embeddings')
    if model_props.use_spans:
        graph.add_input(name='spans', input_shape=(input_sizes['spans'],))

    if model_props.anaphoricity:
        graph.add_input(name='mention_features', input_shape=(input_sizes['mention_features'],))
        add_anaphoricity_reprs(graph, model_props)
        graph.add_node(get_top_layers(model_props, representation), name='top_anaphoricity',
                       input='anaphoricity_reprs')
        if not model_props.ranking:
            graph.add_output(name='anaphoricities', input='top_anaphoricity')
        
    if model_props.anaphoricity_only:
        return graph

    graph.add_input(name='antecedents', input_shape=(1,), dtype='int32')
    graph.add_input(name='pair_antecedents', input_shape=(1,), dtype='int32')
    graph.add_input(name='pair_anaphors', input_shape=(1,), dtype='int32')
    graph.add_input(name='pair_features', input_shape=(input_sizes['pair_features'],))

    add_mention_reprs(graph, 'antecedent', model_props)
    add_mention_reprs(graph, 'anaphor', model_props)

    graph.add_node(Dense(model_props.layer_sizes[0], W_regularizer=model_props.get_regularizer()),
                   name='pair_features_reprs', input='pair_features')
    graph.add_node(get_top_layers(model_props, representation), name='top',
                   inputs=['pair_anaphor_reprs', 'pair_antecedent_reprs', 'pair_features_reprs'],
                   merge_mode='sum')

    if model_props.top_pairs:
        graph.add_input(name='score_inds', input_shape=(1,), dtype='int32')
        graph.add_input(name='starts', input_shape=(1,), dtype='int32')
        graph.add_input(name='ends', input_shape=(1,), dtype='int32')
        graph.add_node(Identity(), name='reordered',
                       inputs=['top', 'score_inds'], merge_mode='index')
        graph.add_node(Identity(), name='maxed_scores',
                       inputs=['reordered', 'starts', 'ends'], merge_mode='imax')
        graph.add_output(name='y', input='maxed_scores')
    elif model_props.ranking:
        graph.add_input(name='reindex', input_shape=(1,), dtype='int32')
        graph.add_input(name='starts', input_shape=(1,), dtype='int32')
        graph.add_input(name='ends', input_shape=(1,), dtype='int32')
        graph.add_input(name='costs', input_shape=(1,))

        if model_props.anaphoricity:
            graph.add_node(Activation(lambda x: -1 - 0.3 * x),
                           name='anaphoricity_scores', input='top_anaphoricity')
            graph.add_node(Identity(), name='concatenated_scores',
                           inputs=['top', 'anaphoricity_scores'], concat_axis=0)
            graph.add_node(Identity(), name='scores_reindexed',
                           inputs=['concatenated_scores', 'reindex'], merge_mode='index')
        else:
            graph.add_node(Identity(), name='scores_reindexed',
                           inputs=['top', 'reindex'], merge_mode='index')
        graph.add_node(Identity(), name='anaphor_losses',
                       inputs=['scores_reindexed', 'starts', 'ends', 'costs'],
                       merge_mode='risk' if model_props.reinforce else 'mm')
        graph.add_output(name='y', input='anaphor_losses')
        graph.add_output(name='z', input='scores_reindexed')
    else:
        graph.add_output(name='y', input='top')
    return graph


def set_weights(graph, weights_from, weights_file):
    print("Setting weights from", weights_from)
    graph.set_weights(get_weights(weights_from, weights_file))


def get_weights(model, weight_file):
    w_file = directories.MODELS + model + '/' + weight_file + '.hdf5'
    print("Loading model '%s' weights '%s' from %s" % (model, weight_file, w_file))
    f = h5py.File(w_file, mode='r')
    g = f['graph']
    return [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]


def get_model(train, vectors, model_props):
    graph = build_graph(train, vectors, model_props)
    opt = model_props.get_optimizer()

    timer.start("compile")
    loss = {}
    if model_props.ranking:
        loss['y'] = get_sum(train.scale_factor * (0.1 if model_props.reinforce else 1))
    else:
        if not model_props.anaphoricity_only:
            loss['y'] = get_summed_cross_entropy(train.scale_factor)
        if model_props.anaphoricity:
            loss['anaphoricities'] = get_summed_cross_entropy(train.anaphoricity_scale_factor)
    graph.compile(loss=loss, optimizer=opt)
    timer.stop("compile")

    if model_props.load_weights_from is not None:
        set_weights(graph, model_props.load_weights_from, model_props.weights_file)

    return graph, opt
