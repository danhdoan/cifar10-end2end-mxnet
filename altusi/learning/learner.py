"""
Learner class
=============
"""


import mxnet as mx
from mxnet import gluon, autograd


class Learner():
    def __init__(self, net, loader, ctx):
        self.net = net
        self.loader = loader
        self.ctx = ctx

        self.loader_iter = iter(self.loader)
        self.net.initialize(mx.init.Xavier(), ctx=self.ctx)
        self.criterion = gluon.loss.SoftmaxCrossEntropyLoss()
        self.trainer = gluon.Trainer(net.collect_params(),
                                     'sgd',
                                     {'learning_rate':0.001})


    def iteration(self, lr=None, take_step=True):
        if lr and lr != self.trainer.load_states:
            self.trainer.set_learning_rate(lr)

        X, y =next(self.loader_iter)
        X, y = X.as_in_context(self.ctx), y.as_in_context(self.ctx)

        with autograd.record():
            y_hat = self.net(X)
            loss = self.criterion(y_hat, y) 

        loss.backward()

        if take_step:
            self.trainer.step(X.shape[0])

        return loss.mean().asscalar()


    def close(self):
        self.loader_iter.shutdown()
