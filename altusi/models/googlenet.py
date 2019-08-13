"""
GoogleNet Network
=================
"""


from mxnet.gluon import nn


class Inception(nn.HybridBlock):
    def __init__(self, nchannels, **kwargs):
        super(Inception, self).__init__(**kwargs)

        c1, c2, c3, c4 = nchannels

        # branch 1
        self.p1 = nn.Conv2D(channels=c1, kernel_size=1, activation='relu')
        
        # branch 2
        self.p2_1 = nn.Conv2D(channels=c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(channels=c2[1], kernel_size=3, padding=1, activation='relu')

        # branch 3
        self.p3_1 = nn.Conv2D(channels=c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(channels=c3[1], kernel_size=5, padding=2, activation='relu')

        # branch 4
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(channels=c4, kernel_size=1, activation='relu')


    def hybrid_forward(self, F, X):
        p1 = self.p1(X)
        p2 = self.p2_2(self.p2_1(X))
        p3 = self.p3_2(self.p3_1(X))
        p4 = self.p4_2(self.p4_1(X))

        return F.concat(p1, p2, p3, p4, dim=1)


class GoogleNet(nn.HybridBlock):
    def __init__(self, nclasses, **kwargs):
        super(GoogleNet, self).__init__(**kwargs)

        self.features = nn.HybridSequential()

        # layer 1
        self.features.add(nn.Conv2D(channels=64, kernel_size=7, 
            strides=2, padding=3, activation='relu'))
        self.features.add(nn.MaxPool2D(pool_size=3, strides=1, padding=1))

        # layer 2
        self.features.add(nn.Conv2D(channels=64, kernel_size=1, activation='relu'))
        self.features.add(nn.Conv2D(channels=192, kernel_size=3, 
                                    padding=1, activation='relu'))
        self.features.add(nn.MaxPool2D(pool_size=3, strides=1, padding=1))

        # inception 3
        self.features.add(Inception([64, (96, 128), (16, 32), 32]))
        self.features.add(Inception([128, (128, 192), (32, 96), 64]))
        self.features.add(nn.MaxPool2D(pool_size=3, strides=1, padding=1))

        # inception 4
        self.features.add(Inception([192, (96, 208), (16, 48), 64]))
        self.features.add(Inception([160, (112, 224), (24, 64), 64]))
        self.features.add(Inception([128, (128, 256), (24, 64), 64]))
        self.features.add(Inception([112, (144, 288), (32, 64), 64]))
        self.features.add(Inception([256, (160, 320), (32, 128), 64]))
        self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))

        # inception 5
        self.features.add(Inception([256, (160, 320), (32, 128), 128]))
        self.features.add(Inception([384, (192, 384), (48, 128), 128]))
        self.features.add(nn.AvgPool2D(pool_size=5, strides=1))
        self.features.add(nn.Dropout(rate=0.4))

        # output
        self.output = nn.Dense(nclasses)


    def hybrid_forward(self, F, X):
        y = self.features(X)
        y = self.output(y)

        return y
