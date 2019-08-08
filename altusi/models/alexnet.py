"""
AlexNet Network
===============
"""


from mxnet.gluon import nn


class AlexNet(nn.HybridBlock):
    def __init__(self, nclasses=10, **kwargs):
        super(AlexNet, self).__init__(**kwargs)

        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(channels=96, kernel_size=11, padding=4))
        self.features.add(nn.BatchNorm())
        self.features.add(nn.Activation('relu'))
        self.features.add(nn.MaxPool2D(pool_size=3, strides=2))

        self.features.add(nn.Conv2D(channels=256, kernel_size=5, padding=2))
        self.features.add(nn.BatchNorm())
        self.features.add(nn.Activation('relu'))
        self.features.add(nn.MaxPool2D(pool_size=3, strides=2))

        for _ in range(3):
            self.features.add(nn.Conv2D(channels=384, kernel_size=3, padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
        self.features.add(nn.MaxPool2D(pool_size=3, strides=2))

        self.features.add(nn.Dense(4096, activation='relu'))
        self.features.add(nn.Dropout(rate=0.5))
        self.features.add(nn.Dense(4096, activation='relu'))
        self.features.add(nn.Dropout(rate=0.5))

        self.output = nn.Dense(nclasses)


    def hybrid_forward(self, F, X):
        y = self.features(X)
        y = self.output(y)

        return y
