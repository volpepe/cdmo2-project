from typing import Dict
import pandas as pd
import numpy as np
import random
from PIL import Image

import sys
import os

sys.path.append(os.path.abspath('.'))

from src.optimizers import Optimizer
from src.graphics import show_animation
from src.actors import ACTORS
from src.constants import DESIRED_SIZE, MAX_HEIGHT, \
    MOUNTAINS, NUM_EPOCHS
    

def generate_steps(actors: Dict[str,Dict], epochs: int) -> None:
    '''
    Generate the steps taken by each actor.
    '''
    for actor in actors:
        optimizer: Optimizer = actors[actor]['optimizer']
        # Obtain starting position for each actor
        start_x,start_y,start_z = optimizer.x, optimizer.y, \
            optimizer.get_z_level(optimizer.x,optimizer.y)
        # Add starting position to actor dictionaries
        actors[actor]['history'] = {
            'epoch': [0],
            'x': [start_x],
            'y': [start_y],
            'z': [start_z]
        }            
        # Override with starting position and start steps generation
        x, y, z = start_x, start_y, start_z
        for i in range(1, epochs):
            # Compute step
            x,y,z = optimizer.next_step()
            # Update dict
            actors[actor]['history']['epoch'].append(i)
            actors[actor]['history']['x'].append(x)
            actors[actor]['history']['y'].append(y)
            actors[actor]['history']['z'].append(z)

if __name__ == '__main__':
    # Use a random mountain
    z_data = pd.read_csv(MOUNTAINS[random.randint(0, len(MOUNTAINS)-1)])
    # If needed resize the data by interpolation
    z_array = np.array(Image.fromarray(z_data.to_numpy(dtype=np.float32)).resize(
        (DESIRED_SIZE, DESIRED_SIZE), resample=Image.BILINEAR)
    )
    # Recreate dataframe from rescaled features (range [0,MAX_HEIGHT])
    z_data = pd.DataFrame(z_array / np.max(z_array) * MAX_HEIGHT)
    # Initialize optimizers z_data
    for actor in ACTORS:
        ACTORS[actor]['optimizer'] = ACTORS[actor]['optimizer'](z_data)
    # Compute steps for actors
    generate_steps(ACTORS, NUM_EPOCHS)
    # Produce and show final animation
    show_animation(z_data, ACTORS)