from optimizers import *

ACTORS = {
    'alice': {
        'optimizer': NelderMeadOptimizer,
        'optimizer_type': 'nelden_mead_optimizer'
    },
    'bob': {
        'optimizer': RandomOptimizer,
        'optimizer_type': 'random_optimizer'
    },
    'carl': {
        'optimizer': RandomOptimizer,
        'optimizer_type': 'random_optimizer'
    },
    'david': {
        'optimizer': RandomOptimizer,
        'optimizer_type': 'random_optimizer'
    },
    'ed': {
        'optimizer': RandomOptimizer,
        'optimizer_type': 'random_optimizer'
    }
}