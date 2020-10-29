import directories
import pairwise_models
import model_properties
from custom_neural_implementations import Identity, max_margin, risk

import numpy as np
from keras.models import Graph
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam, SGD


def pair_weights(weights):
    return [
        np.vstack([np.asarray(weights[i]) for i in range(15, 25, 2)]),
        np.sum(np.vstack([np.asarray(weights[i]) for i in range(16, 26, 2)]), axis=0),
    ] + [np.asarray(w) for w in weights[25:33]]


def anaphoricity_weights(weights):
    return [
        np.vstack([np.asarray(weights[i]) for i in range(1, 7, 2)]),
        np.sum(np.vstack([np.asarray(weights[i]) for i in range(2, 8, 2)]), axis=0),
    ] + [np.asarray(w) for w in weights[7:15]]


def get_weights(model_props):
    weights = pairwise_models.get_weights(model_props.load_weights_from, model_props.weights_file)
    w_ana = anaphoricity_weights(weights)
    w_pair = pair_weights(weights)

    static_ana = w_ana[:2 * model_props.static_layers]
    dynamic_ana = w_ana[2 * model_props.static_layers:]
    dynamic_ana[-2] *= -0.3
    dynamic_ana[-1] -= 1
    static_pair = w_pair[:2 * model_props.static_layers]
    dynamic_pair = w_pair[2 * model_props.static_layers:-4]

    Wscore, bscore = w_pair[-4], w_pair[-3]
    Wscale, bscale = w_pair[-2], w_pair[-1]
    Wscore = np.vstack([Wscore for _ in range(2 if model_props.pooling == 'maxavg' else 1)]) + \
             (0.1 * (np.random.random((2 * Wscore.shape[0], Wscore.shape[1])) - 0.5))
    Wscore[Wscore.shape[0]/2, :] *= 0.15
    dynamic_pair += [Wscore, bscore, Wscale, bscale]
    return static_ana, dynamic_ana, static_pair, dynamic_pair, np.asarray(weights[0])


def get_static_model(model_props, anaphoricity=False):
    if model_props.static_layers == 0:
        return None
    graph = Graph()
    graph.add_input(name='X', input_shape=(model_props.single_size if anaphoricity else
                                           model_props.pair_size,))
    graph.add_node(Dense(1000, activation='relu'), name='h1', input='X')
    graph.add_node(Dropout(0.5), name='h1drop', input='h1')
    if model_props.static_layers > 1:
        graph.add_node(Dense(500, activation='relu'), name='h2', input='h1drop')
        graph.add_node(Dropout(0.5), name='h2drop', input='h2')
    if model_props.static_layers > 2:
        graph.add_node(Dense(500, activation='relu'), name='h3', input='h2drop')
        graph.add_node(Dropout(0.5), name='h3drop', input='h3')
    graph.add_output('out', input='h' + str(min(model_props.top_layers,
                                                model_props.static_layers)) + 'drop')
    print("Compiling {:} model".format("anaphoricity" if anaphoricity else "pair"))
    graph.compile(loss={'out': 'mse'}, optimizer=SGD())
    return graph


def add_model(graph, prefix, features_name, model_props):
    graph.add_node(Dropout(model_props.input_dropout), name=features_name + '_dropped',
                   input=features_name)
    current_layer = features_name + '_dropped'
    if model_props.learnable_layers == 3:
        graph.add_node(Dense(1000, activation='relu'), name=prefix + 'h1', input=current_layer)
        graph.add_node(Dropout(model_props.dropout), name=prefix + 'h1drop', input=prefix + 'h1')
        current_layer = prefix + 'h1drop'
    if model_props.learnable_layers >= 2:
        graph.add_node(Dense(500, activation='relu'), name=prefix + 'h2', input=current_layer)
        graph.add_node(Dropout(model_props.dropout), name=prefix + 'h2drop', input=prefix + 'h2')
        current_layer = prefix + 'h2drop'
    if model_props.learnable_layers >= 1:
        graph.add_node(Dense(500, activation='relu'), name=prefix + 'h3', input=current_layer)
        graph.add_node(Dropout(model_props.dropout), name=prefix + 'h3drop', input=prefix + 'h3')
        current_layer = prefix + 'h3drop'
    return current_layer


def get_model(model_props):
    graph = Graph()
    graph.add_input(name='pair_features', input_shape=(model_props.pair_input_size,))
    graph.add_input(name='mention_features', input_shape=(model_props.anaphoricity_input_size,))
    graph.add_input(name='starts', input_shape=(1,), dtype='int32')
    graph.add_input(name='ends', input_shape=(1,), dtype='int32')

    anaphoricity_repr = add_model(graph, 'anaphoricity', 'mention_features', model_props)
    graph.add_node(Dense(1), name='anaphoricity_score', input=anaphoricity_repr)
    graph.add_node(Dense(1), name='scaled_anaphoricity_score', input='anaphoricity_score')

    pair_repr = add_model(graph, '', 'pair_features', model_props)
    graph.add_node(Identity(), name='cluster_reprs', inputs=[pair_repr, 'starts', 'ends'],
                   merge_mode='i' + model_props.pooling)
    graph.add_node(Dense(1), name='merge_scores', input='cluster_reprs')
    graph.add_node(Dense(1), name='scaled_merge_scores', input='merge_scores')
    graph.add_node(Identity(), name='action_scores',
                   inputs=['scaled_merge_scores', 'scaled_anaphoricity_score'], concat_axis=0)
    graph.add_output(name='costs', input='action_scores')

    opt = Adam(lr=model_props.learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-6)
    print("Compiling clustering model")
    graph.compile(loss={'costs': risk if model_props.risk_objective else max_margin}, optimizer=opt)
    return graph, opt


def get_models(model_props):
    anaphoricity_static = get_static_model(model_props, True)
    pair_static = get_static_model(model_props, False)
    clusterer, opt = get_model(model_props)
    static_ana, dynamic_ana, static_pair, dynamic_pair, word_vectors = get_weights(model_props)

    if pair_static is not None:
        pair_static.set_weights(static_pair)
        anaphoricity_static.set_weights(static_ana)
    print([w.shape for w in dynamic_pair])
    if not model_props.randomize_weights:
        clusterer.set_weights(dynamic_ana + dynamic_pair)
    return pair_static, anaphoricity_static, clusterer, word_vectors
