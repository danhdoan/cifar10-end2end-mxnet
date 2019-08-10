"""
DenseNet Network
================
"""

from mxnet import nd
from mxnet.gluon import nn


def _make_dense_layer(growth_rate, bn_size):
    layer = nn.HybridSequential()

    layer.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(bn_size * growth_rate, kernel_size=1, use_bias=False),

        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(growth_rate, kernel_size=3, padding=1, use_bias=False)
    )

    return layer


class DenseBlock(nn.HybridBlock):
    def __init__(self, nconvs, growth_rate, bn_size, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)

        self.features = nn.HybridSequential()
        for _ in range(nconvs):
            self.features.add(_make_dense_layer(growth_rate, bn_size))


    def hybrid_forward(self, F, X):
        for layer in self.features:
            y = layer(X)
            X = F.concat(X, y, dim=1)

        return y


def _make_transition_layer(nout_channels):
    layer = nn.HybridSequential()
    
    layer.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(nout_channels, kernel_size=1, use_bias=False),
        nn.MaxPool2D(pool_size=2, strides=2)
    )
    
    return layer


class DenseNet(nn.HybridBlock):
    def __init__(self, nin_channels, growth_rate, nconvs_blocks, nclasses=1000,
                 bn_size=2, **kwargs):
        super(DenseNet, self).__init__(**kwargs)

        self.features = nn.HybridSequential()

        # add first CONV layer
        self.features.add(nn.Conv2D(nin_channels, kernel_size=7,
                                   strides=1, padding=3, use_bias=False))
        self.features.add(nn.BatchNorm())
        self.features.add(nn.Activation('relu'))

        # add POOL layer
        self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))

        # add DenseBlock
        for i, nconvs in enumerate(nconvs_blocks):
            self.features.add(DenseBlock(nconvs, growth_rate, bn_size))
            nin_channels += nconvs * growth_rate

            # add Transition layer
            if i+1 < len(nconvs_blocks):
                nin_channels //= 2
                self.features.add(_make_transition_layer(nin_channels))

        self.features.add(nn.BatchNorm())
        self.features.add(nn.Activation('relu'))
        self.features.add(nn.AvgPool2D(pool_size=2))

        self.output = nn.Dense(nclasses)


    def hybrid_forward(self, F, X):
        X = self.features(X)
        X = self.output(X)

        return X


densenet_archs = {
    121: [64, 32, [6, 12, 24, 16]],
    161: [96, 48, [6, 12, 36, 24]],
    169: [64, 32, [6, 12, 32, 32]],
    201: [64, 32, [6, 12, 48, 32]]
}


def DenseNet121(nclasses):
    return DenseNet(*densenet_archs[121], nclasses)


def DenseNet161(nclasses):
    return DenseNet(*densenet_archs[161], nclasses)


def DenseNet169(nclasses):
    return DenseNet(*densenet_archs[169], nclasses)


def DenseNet201(nclasses):
    return DenseNet(*densenet_archs[201], nclasses)
