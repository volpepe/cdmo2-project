import pandas as pd
from typing import Dict, Any
import plotly.graph_objects as go
import plotly.express as px

def show_animation( z_data: pd.DataFrame, actors: Dict[str, Dict], 
                    title:str='Snowboarders',
                    width: int = 1200, height: int = 720, 
                    margin: Dict[Any,int] = dict(l=100, r=100, b=100, t=100)) -> None:
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
        df['z'] = df['z'].apply(lambda x: x)
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
