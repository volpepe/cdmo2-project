from optimizers import *

ACTORS = {
    'alice': {
        'optimizer': RandomOptimizer,
        'optimizer_type': 'random_optimizer'
    },
    'bob': {
        'optimizer': BacktrackingLineSearchOptimizer,
        'optimizer_type': 'backtracking_line_search_optimizer'
    },
    'carl': {
        'optimizer': NelderMeadOptimizer,
        'optimizer_type': 'nelder_mead_optimizer'
    },
    'david': {
        'optimizer': ParticleSwarmOptimizer,
        'optimizer_type': 'particle_swarm_optimizer'
    }
}