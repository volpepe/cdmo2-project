from typing import Dict, Tuple
import pandas as pd
import numpy as np
import math
import random
from PIL import Image

import sys
import os
sys.path.append(os.path.abspath('.'))

from src.graphics import show_animation
from src.constants import ACTORS, DESIRED_SIZE, \
    MOUNTAINS, NUM_EPOCHS


def next_step(x:int,y:int,z_map:pd.DataFrame):
    shape = z_map.shape
    move_x, move_y = random.randint(-1,1), random.randint(-1,1)
    # Apply update and index on z_map. Note that on z_map array 
    # y refers to the row and x to the column
    n_x, n_y = (x+move_x) % shape[0], (y+move_y) % shape[1]
    n_z = z_map.iloc[n_y, n_x]
    return (n_x, n_y, n_z)


def get_starting_point(z_data:pd.DataFrame) -> Tuple:
    '''
    Computes the starting point of the actors (the position with maximum
    height on the map).
    '''
    z_arr = np.array(z_data)    
    col = np.argmax(z_arr) % z_arr.shape[1]
    row = math.floor(np.argmax(z_arr) / z_arr.shape[1])
    return (col, row, z_arr[row,col])


def generate_steps(actors: Dict[str,Dict], z_data: pd.DataFrame, epochs: int) -> None:
    '''
    Generate the steps taken by each actor
    '''
    # Obtain starting position for each actor
    start_x,start_y,start_z = get_starting_point(z_data)

    for actor in actors:
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
            x,y,z = next_step(x,y,z_data)
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
    # Recreate dataframe from rescaled features (range [0,DESIRED_SIZE])
    z_data = pd.DataFrame(z_array / np.max(z_array) * DESIRED_SIZE)
    # Compute steps for actors
    generate_steps(ACTORS, z_data, NUM_EPOCHS)
    # Produce and show final animation
    show_animation(z_data, ACTORS)