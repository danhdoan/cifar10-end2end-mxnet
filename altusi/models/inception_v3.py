"""
InceptionV3 Network
===================
"""


from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import HybridConcurrent


def _make_conv(**kwargs):
    block = nn.HybridSequential()

    block.add(nn.Conv2D(use_bias=False, **kwargs))
    block.add(nn.BatchNorm(epsilon=0.001))
    block.add(nn.Activation('relu'))

    return block


def _make_branch(use_pool, *conv_settings):
    block = nn.HybridSequential()

    if use_pool == 'avg':
        block.add(nn.AvgPool2D(pool_size=3, strides=1, padding=1))
    elif use_pool == 'max':
        block.add(nn.MaxPool2D(pool_size=3, strides=2))

    setting_names = ['channels', 'kernel_size', 'strides', 'padding']
    for setting in conv_settings:
        kwargs = {}
        for i, value in enumerate(setting):
            if value is not None:
                kwargs[setting_names[i]] = value

        block.add(_make_conv(**kwargs))

    return block


def _make_A(pool_features):
    block = HybridConcurrent(axis=1)
    
    block.add(_make_branch(None, 
                           (64, 1, None, None)))
    block.add(_make_branch(None, 
                           (48, 1, None, None),
                           (64, 5, None, 2)))
    block.add(_make_branch(None,
                           (64, 1, None, None),
                           (96, 3, None, 1),
                           (96, 3, None, 1)))
    block.add(_make_branch('avg',
                           (pool_features, 1, None, None)))

    return block


def _make_B():
    block = HybridConcurrent(axis=1)

    block.add(_make_branch(None, 
                           (384, 3, 2, None)))
    block.add(_make_branch(None,
                           (64, 1, None, None),
                           (96, 3, None, 1),
                           (96, 3, 2, None)))
    block.add(_make_branch('max'))

    return block


def _make_C(channels):
    block = HybridConcurrent(axis=1)

    block.add(_make_branch(None,
                           (192, 1, None, None)))
    block.add(_make_branch(None,
                           (channels, 1, None, None),
                           (channels, (1, 5), None, (0, 2)),
                           (192, (5, 1), None, (2, 0))))

    block.add(_make_branch(None,
                           (channels, 1, None, None),
                           (channels, (5, 1), None, (2, 0)),
                           (channels, (1, 5), None, (0, 2)),
                           (channels, (5, 1), None, (2, 0)),
                           (192, (1, 5), None, (0, 2))))

    block.add(_make_branch('avg',
                           (192, 1, None, None)))

    return block


def _make_D():
    block = HybridConcurrent(axis=1)

    block.add(_make_branch(None,
                           (192, 1, None, None),
                           (320, 3, 2, None)))

    block.add(_make_branch(None,
                           (192, 1, None, None),
                           (192, (1, 5), None, (0, 2)),
                           (192, (5, 1), None, (2, 0)),
                           (192, 3, 2, None)))

    return block


def _make_E():
    block = HybridConcurrent(axis=1)
    block.add(_make_branch(None,
                         (320, 1, None, None)))

    branch_3x3 = nn.HybridSequential()
    block.add(branch_3x3)
    branch_3x3.add(_make_branch(None,
                                (384, 1, None, None)))
    branch_3x3_split = HybridConcurrent(axis=1, prefix='')
    branch_3x3_split.add(_make_branch(None,
                                      (384, (1, 3), None, (0, 1))))
    branch_3x3_split.add(_make_branch(None,
                                      (384, (3, 1), None, (1, 0))))
    branch_3x3.add(branch_3x3_split)

    branch_3x3dbl = nn.HybridSequential(prefix='')
    block.add(branch_3x3dbl)
    branch_3x3dbl.add(_make_branch(None,
                                   (448, 1, None, None),
                                   (384, 3, None, 1)))
    branch_3x3dbl_split = HybridConcurrent(axis=1, prefix='')
    branch_3x3dbl.add(branch_3x3dbl_split)
    branch_3x3dbl_split.add(_make_branch(None,
                                         (384, (1, 3), None, (0, 1))))
    branch_3x3dbl_split.add(_make_branch(None,
                                         (384, (3, 1), None, (1, 0))))

    block.add(_make_branch('avg',
                         (192, 1, None, None)))

    return block


class InceptionV3(nn.HybridBlock):
    def __init__(self, nclasses=10, **kwargs):
        super(InceptionV3, self).__init__(**kwargs)

        self.features = nn.HybridSequential()

        # pre-liminary layers
        self.features.add(_make_conv(channels=32, kernel_size=3, strides=1, padding=1))
        self.features.add(_make_conv(channels=32, kernel_size=3, padding=1))
        self.features.add(_make_conv(channels=64, kernel_size=3, padding=1))

        self.features.add(nn.MaxPool2D(pool_size=3, strides=1))

        self.features.add(_make_conv(channels=80, kernel_size=1))
        self.features.add(_make_conv(channels=192, kernel_size=3, strides=1, padding=1))
        self.features.add(nn.MaxPool2D(pool_size=3, strides=1))

        # make A
        self.features.add(_make_A(32))
        self.features.add(_make_A(64))
        self.features.add(_make_A(64))

        # make B
        self.features.add(_make_B())

        # make C
        self.features.add(_make_C(128))
        self.features.add(_make_C(160))
        self.features.add(_make_C(160))
        self.features.add(_make_C(192))

        # make D
        self.features.add(_make_D())

        # make E
        self.features.add(_make_E())
        self.features.add(_make_E())

        self.features.add(nn.AvgPool2D(pool_size=4))
        self.features.add(nn.Dropout(rate=0.5))

        # blockput
        self.output = nn.Dense(nclasses)


    def hybrid_forward(self, F, X):
        y = self.features(X)
        y = self.output(y)

        return y
