from typing import Dict, Tuple
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
from src.constants import *

def get_extension(filename:str) -> str:
    '''
    Returns file extension.
    '''
    return filename.split('.')[-1]

def open_file(filename:str) -> Tuple[np.array, pd.DataFrame]:
    '''
    Opens a tif or csv file containing a mountain (represented as a 2D array).

    Returns a numpy array containing the heights and a reduced Dataframe for visualization.
    '''
    ext = get_extension(filename)

    if ext == 'csv':
        z_im = Image.fromarray(
            pd.read_csv(filename).to_numpy(dtype=np.float32)
        )
    elif ext == 'tif':
        z_im = Image.open(filename)

    # Reshape images to the common format
    z_array = np.array(z_im.resize((DATA_W, DATA_H)))
    # Create a dataframe from rescaled features (range [0,MAX_HEIGHT])
    z_data = z_array / np.max(z_array) * MAX_HEIGHT
    # Resize the data by interpolation to obtain a grid that can
    # be displayed more easily (also height-rescaled)
    visual_grid = pd.DataFrame(np.array(z_im.resize(
        (VISUAL_W, VISUAL_H), resample=Image.BILINEAR)
    ) / np.max(z_array) * MAX_HEIGHT)

    return z_data, visual_grid

def generate_steps(actors: Dict[str,Dict], epochs: int) -> None:
    '''
    Generate the steps taken by each actor.
    '''
    for actor in actors:
        optimizer: Optimizer = actors[actor]['optimizer']
        if actors[actor]['optimizer_type'] == 'particle_swarm_optimizer':
            # Obtain starting position for each particle (in this particular optimizer, they are lists)
            start_x,start_y,start_z = optimizer.x, optimizer.y, \
                [ optimizer.get_z_level(*p.p) for p in optimizer.particles ]
        else: 
            # Obtain starting position for actor
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
        # At the end, print the best position found by the actor
        if actors[actor]['optimizer_type'] == 'particle_swarm_optimizer':
            print("After {} epochs, the algorithm {} is at ({:.2f}, {:.2f}), height {}.".format(
                epochs, actors[actor]['optimizer_type'],
                        np.average(actors[actor]['history']['x'][-1]), 
                        np.average(actors[actor]['history']['y'][-1]),
                        np.average(actors[actor]['history']['z'][-1])
            ))
        else:
            print("After {} epochs, the algorithm {} is at ({:.2f}, {:.2f}), height {}.".format(
                epochs, actors[actor]['optimizer_type'],
                        actors[actor]['history']['x'][-1], 
                        actors[actor]['history']['y'][-1],
                        actors[actor]['history']['z'][-1]
            ))


if __name__ == '__main__':
    # Open and process a random mountain
    mountain = MOUNTAINS[random.randint(0, len(MOUNTAINS)-1)]
    z_data, visual_data = open_file(mountain)
    # Initialize optimizers z_data
    for actor in ACTORS:
        ACTORS[actor]['optimizer'] = ACTORS[actor]['optimizer'](z_data, STARTING_POS_AREA)
    # Compute steps for actors
    generate_steps(ACTORS, NUM_EPOCHS)
    # Print best position for comparison
    loc_max = np.argmax(z_data)
    y_max = loc_max // DATA_W
    x_max = loc_max - y_max*DATA_W
    print("The best position is at ({}, {}), height {}".format(
        x_max, y_max, z_data[y_max, x_max]
    ))
    # Produce and show final animation
    show_animation(visual_data, ACTORS)