"""
LR Scheduler
============
"""


import math
import copy
import mxnet as mx


class TriangularScheduler():
    def __init__(self, min_lr, max_lr, cycle_length, inc_fraction=0.5):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.inc_fraction = inc_fraction


    def __call__(self, iteration):
        if iteration <= int(self.cycle_length * self.inc_fraction):
            unit_cycle = iteration / (self.cycle_length * self.inc_fraction)
        elif iteration <= self.cycle_length:
            unit_cycle = (self.cycle_length - iteration) / (self.cycle_length * (1 - self.inc_fraction))
        else:
            unit_cycle = 0

        adjusted_cycle = unit_cycle * (self.max_lr - self.min_lr) + self.min_lr

        return adjusted_cycle


class LinearWarmUp():
    def __init__(self, schedule, start_lr, length):
        self.schedule = schedule
        self.start_lr = start_lr
        self.finish_lr = copy.copy(schedule)(0)
        self.length = length


    def __call__(self, iteration):
        if iteration <= self.length:
            return iteration * (self.finish_lr - self.start_lr) / self.length + self.start_lr
        else:
            return self.schedule(iteration - self.length)


class LinearCoolDown():
    def __init__(self, schedule, finish_lr, start_idx, length):
        self.schedule = schedule
        self.start_lr = copy.copy(self.schedule)(start_idx)
        self.finish_lr = finish_lr
        self.start_idx = start_idx
        self.finish_idx = start_idx + length
        self.length = length


    def __call__(self, iteration):
        if iteration <= self.start_idx:
            return self.schedule(iteration)
        elif iteration <= self.finish_idx:
            return (iteration - self.start_idx) * (self.finish_lr - self.start_lr) / \
                self.length + self.start_lr
        else:
            return self.finish_lr


class OneCycleScheduler():
    def __init__(self, start_lr, max_lr, cycle_length, cooldown_length=0, finish_lr=None):
        finish_lr = finish_lr if cooldown_length > 0 else start_lr

        schedule = TriangularScheduler(min_lr=start_lr,
                                       max_lr=max_lr,
                                       cycle_length=cycle_length)
        self.schedule = LinearCoolDown(schedule, finish_lr=finish_lr, 
                                       start_idx=cycle_length,
                                       length=cooldown_length)

    def __call__(self, iteration):
        return self.schedule(iteration)


class CyclicalSchedule():
    def __init__(self, schedule_class, cycle_length, cycle_length_decay=1, cycle_magnitude_decay=1, **kwargs):
        self.schedule_class = schedule_class
        self.length = cycle_length
        self.length_decay = cycle_length_decay
        self.magnitude_decay = cycle_magnitude_decay
        self.kwargs = kwargs
    
    def __call__(self, iteration):
        cycle_idx = 0
        cycle_length = self.length
        idx = self.length
        while idx <= iteration:
            cycle_length = math.ceil(cycle_length * self.length_decay)
            cycle_idx += 1
            idx += cycle_length
        cycle_offset = iteration - idx + cycle_length
        
        schedule = self.schedule_class(max_update=cycle_length, **self.kwargs)
        return schedule(cycle_offset) * self.magnitude_decay**cycle_idx
