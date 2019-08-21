"""
Learning Rate Finder
====================
"""


import altusi.utils.visualizer as vis


class LRFinderStoppingCriteria():
    def __init__(self, smoothing=0.3, min_iter=20):
        self.smoothing = smoothing
        self.min_iter = min_iter
        self.first_loss = None
        self.moving_mean = None
        self.cnter = 0

    
    def __call__(self, loss):
        self.cnter += 1

        if self.first_loss is None:
            self.first_loss = self.moving_mean = loss
        else:
            self.moving_mean = (1 - self.smoothing)*loss + \
                self.smoothing * self.moving_mean

        return self.moving_mean > self.first_loss * 2 and self.cnter >= self.min_iter


class LRFinder():
    def __init__(self, learner):
        self.learner = learner


    def find(self, lr_start=1e-6, lr_multiplier=1.1, smoothing=0.3):
        lr = lr_start

        self.results = []
        stopping_criteria = LRFinderStoppingCriteria(smoothing=smoothing,
                                                     min_iter=20)

        while True:
            loss = self.learner.iteration(lr)

            self.results.append((lr, loss))

            if stopping_criteria(loss):
                break

            lr *= lr_multiplier

        return self.results


    def plot(self, figsize=(8, 6)):
        lrs = [result[0] for result in self.results]
        losses = [result[1] for result in self.results]

        min_idx = 0
        for i, loss in enumerate(losses):
            if loss < losses[min_idx]:
                min_idx = i

        min_y = losses[min_idx]
        avg_y = sum(losses[:min_idx+1]) / (min_idx+1)

        vis.plot(lrs, losses, title='LR Finder',
                 xlabel='lr', ylabel='loss',
                 xscale='log', figsize=figsize, ylim=[min_y, avg_y*1.1])

