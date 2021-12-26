from typing import Any, Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import math
import random

MOUNTAINS = [
    'https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv',
    'https://raw.githubusercontent.com/plotly/datasets/master/volcano.csv'
]

ACTORS = {
    'alice': {
    },
    'bob': {
    },
    'carl': {
    },
    'david': {
    },
    'ed': {
    }
}

NUM_EPOCHS = 100


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


def show_animation( z_data: pd.DataFrame, actors: Dict[str, Dict], 
                    title:str='Snowboarders',
                    width: int = 1200, height: int = 720, 
                    margin: Dict[Any,int] = dict(l=65, r=50, b=65, t=90)) -> None:
    '''
    Creates the map and animation visualization and shows them.
    '''
    # Draw actors
    actors_dfs = []

    for actor in actors:
        # Create a dataframe from the history of the actor's position
        df = pd.DataFrame.from_dict(actors[actor]['history'])
        # Add actor name to the dataframe
        df['actor'] = actor
        df['size'] = 10
        # Append to list to stack later
        actors_dfs.append(df)
    
    # Concatenate all dictionaries
    actors_df = pd.concat(actors_dfs)

    # Create an animated 3D scatter plot
    fig = px.scatter_3d(actors_df, x='x', y='y', z='z',
        color='actor', animation_group='actor',
        animation_frame='epoch', hover_name='actor', 
        size='size', size_max=25
    )
    
    # Add map background + contours on top or bottom axis
    fig = fig.add_trace(
        go.Surface(
            z=z_data,
            colorbar_x=-0.1,
            contours_z=dict(
                show=True, usecolormap=True, 
                highlightcolor="limegreen", 
                project_z=True
            )
        )
    )

    # Finally, display the whole figure
    fig.update_layout(title=title, autosize=False,
                        width=width, height=height,
                        margin=margin).show()


if __name__ == '__main__':
    # Use a random mountain
    z_data = pd.read_csv(MOUNTAINS[random.randint(0, len(MOUNTAINS)-1)])
    # Compute steps for actors
    generate_steps(ACTORS, z_data, NUM_EPOCHS)
    # Produce and show final animation
    show_animation(z_data, ACTORS)