import os


# =============================================================================
# PROJECT'S ORGANIZATION
# =============================================================================
PROJECT_BASE = '.'
CHECKPOINTS = os.path.join(PROJECT_BASE, 'checkpoints')
HISTORY = os.path.join(PROJECT_BASE, 'history')


# =============================================================================
# CIFAR10 DATASET
# =============================================================================
IMAGE_SIZE = 32
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

CIFAR_NCLASSES = 10
CIFAR_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
                 'dog', 'frog', 'horse', 'ship', 'truck')


# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
BATCH_SIZE = 64 

LR = 0.1
LR_DECAY_FACTOR = 0.1
WD = 0.0005
MOMENTUM = 0.9
NEPOCHS = 100
