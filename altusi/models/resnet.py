"""
ResNet Network
==============
"""


from mxnet import nd
from mxnet.gluon import nn


class BasicBlock(nn.HybridBlock):
    def __init__(self, nchannels, downsample=False, strides=1, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        
        self.body = nn.HybridSequential()
        self.body.add(nn.Conv2D(nchannels, kernel_size=3, strides=strides, padding=1))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        
        self.body.add(nn.Conv2D(nchannels, kernel_size=3, padding=1))
        self.body.add(nn.BatchNorm())
        
        if downsample:
            self.shortcut = nn.Conv2D(nchannels, kernel_size=1, strides=strides)
        else:
            self.shortcut = None
            
            
    def hybrid_forward(self, F, X):
        y = self.body(X)
        
        if self.shortcut:
            X = self.shortcut(X)
            
        return F.Activation(y + X, act_type='relu')


class BottleNeck(nn.HybridBlock):
    def __init__(self, nchannels, downsample=False, strides=1, **kwargs):
        super(BottleNeck, self).__init__(**kwargs)
        
        self.body = nn.HybridSequential()
        self.body.add(nn.Conv2D(nchannels//4, kernel_size=1, strides=strides))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        
        self.body.add(nn.Conv2D(nchannels//4, kernel_size=3, strides=1, padding=1))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        
        self.body.add(nn.Conv2D(nchannels, kernel_size=1))
        self.body.add(nn.BatchNorm())
        
        if downsample:
            self.shortcut = nn.Conv2D(nchannels, kernel_size=1, strides=strides)
        else:
            self.shortcut = None
            
    
    def hybrid_forward(self, F, X):
        y = self.body(X)
        
        if self.shortcut:
            X = self.shortcut(X)
            
        return F.Activation(y + X, act_type='relu')


def _make_layer(block, nblocks, nchannels, strides, in_channels):
    layer = nn.HybridSequential()
    
    layer.add(block(nchannels, nchannels != in_channels, strides))
    for _ in range(nblocks-1):
        layer.add(block(nchannels, False))
        
    return layer


def get_resnet(block, layers, channels, nclasses):
    net = nn.HybridSequential()

    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1))

    for i, nblocks in enumerate(layers):
        strides = 1 if i == 0 else 2
        net.add(_make_layer(block, nblocks, channels[i+1], strides, channels[i]))

    net.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.GlobalAvgPool2D(),
            nn.Dense(nclasses))

    return net


resnet_archs = {
    18:  [BasicBlock, [2, 2, 2, 2],  [64,  64, 128,  256,  512]],
    34:  [BasicBlock, [3, 4, 6, 3],  [64,  64, 128,  256,  512]],
    50:  [BottleNeck, [3, 4, 6, 3],  [64, 256, 512, 1024, 2048]],
    101: [BottleNeck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048]],
    152: [BottleNeck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048]]
}


def ResNet18(nclasses):
    return get_resnet(*resnet_archs[18], nclasses)


def ResNet34(nclasses):
    return get_resnet(*resnet_archs[34], nclasses)


def ResNet50(nclasses):
    return get_resnet(*resnet_archs[50], nclasses)


def ResNet101(nclasses):
    return get_resnet(*resnet_archs[101], nclasses)


def ResNet152(nclasses):
    return get_resnet(*resnet_archs[152], nclasses)
