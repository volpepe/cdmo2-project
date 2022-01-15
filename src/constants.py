MOUNTAINS = [
    #'https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv',
    #'https://raw.githubusercontent.com/plotly/datasets/master/volcano.csv',
    'data/mountains.tif'
]
NUM_EPOCHS = 100
MAX_HEIGHT = 100
DATA_H = 4335
DATA_W = 5167
VISUAL_H = 434 # /10
VISUAL_W = 517 # /10

STARTING_POS_AREA = 10      # Start within 10-percentile of height
RANDOM_MOVEMENT_RANGE = 400 # The range of movement for actors that move randomly
NELDER_MEAD_C = 700         # c parameter in Nelder Mead (side size of initial simplex)
LINE_SEARCH_H = 20          # The range for computing the gradient approximation in Line Search
LINE_SEARCH_START_A = 20000 # The starting step size which is iteratively decreased by Armijo
PSO_V0_SCALE = 200          # The initial velocity of particles in PSO
PSO_INERTIA = 0.8           # The initial inertia in PSO