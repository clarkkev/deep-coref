from keras.layers.core import Layer
import theano.tensor as T


class Identity(Layer):
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        return X

    def get_config(self):
        config = {"name": self.__class__.__name__}
        base_config = super(Identity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def max_margin(costs, scores):
    return T.max(costs * (2 + scores - T.max(scores[T.eq(costs, 0).nonzero()[0]])))


def risk(costs, scores):
    e_x = T.exp(scores - scores.max(axis=0, keepdims=True))
    e_x = e_x / e_x.sum(axis=0, keepdims=True)
    return T.sum(costs * e_x)


def get_summed_cross_entropy(n):
    def summed_cross_entropy(y_true, y_pred):
        epsilon = 1.0e-7
        y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
        bce = T.nnet.binary_crossentropy(y_pred, y_true).sum()
        return bce / n
    return summed_cross_entropy


def get_sum(n):
    def summed(y_true, y_pred):
        return (y_true.sum() + y_pred.sum()) / n
    return summed
