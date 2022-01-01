from optimizers import *

ACTORS = {
    'alice': {
        'optimizer': NelderMeadOptimizer,
        'optimizer_type': 'nelden_mead_optimizer'
    },
    'bob': {
        'optimizer': BacktrackingLineSearchOptimizer,
        'optimizer_type': 'backtracking_line_search_optimizer'
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