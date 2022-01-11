import pandas as pd
from typing import Dict, Any
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
# from plotly.subplots import make_subplots

from src.constants import DATA_H, DATA_W, VISUAL_H, VISUAL_W

pio.renderers.default = 'browser'

def show_animation( visual_data: pd.DataFrame,
                    actors: Dict[str, Dict], 
                    title:str='Mountain Climbers',
                    width: int = 1200, height: int = 720, 
                    margin: Dict[Any,int] = dict(l=100, r=100, b=100, t=100, autoexpand=False)) -> None:
    '''
    Creates the map and animation visualization and shows them.
    '''
    # fig = make_subplots(rows=1, cols=2, 
    #     specs=[[{"type": "scene"}, {"type": "xy"}]],)

    # # Draw surface using the smaller dataframe
    # fig.add_trace(
    #     go.Surface(
    #         z=visual_data,
    #         colorbar_x=-0.1,
    #         contours_z=dict(
    #             show=True, usecolormap=True, 
    #             highlightcolor="limegreen", 
    #             project_z=True
    #         )
    #     ),
    #     row=1, col=1
    # )   

    # # Draw actors on the contour plot
    actors_dfs = []

    for actor in actors:
        # Create a dataframe from the history of the actor's position
        if actors[actor]['optimizer_type'] == 'particle_swarm_optimizer':
            df = pd.DataFrame.from_dict(actors[actor]['history'])
            # Create a dataframe by exploding all lists of particle positions
            x = df.explode('x')[['epoch', 'x']]
            y = df.explode('y')['y']
            z = df.explode('z')['z']
            x['y'] = y
            x['z'] = z
            x.reset_index(drop=True)
            df = x.copy()
            df['size'] = 2
        else:
            df = pd.DataFrame.from_dict(actors[actor]['history'])
            df['size'] = 10
        # Scale x/y coordinates for visuals
        df['x'] = df['x'] * VISUAL_W / DATA_W
        df['y'] = df['y'] * VISUAL_H / DATA_H
        df['z'] = df['z'].apply(lambda x: x + 1) # add 1 unit for visualization
        # Add actor name to the dataframe
        df['actor'] = actor
        # Add size and optimizer name
        df['optimizer'] = actors[actor]['optimizer_type']
        # Append to list to stack later
        actors_dfs.append(df)
    
    # Concatenate all dictionaries
    actors_df = pd.concat(actors_dfs, sort=False)

    # Create an animated 3D scatter plot
    fig = px.scatter_3d(actors_df, x='x', y='y', z='z',
        color='actor', animation_group='actor',
        animation_frame='epoch', hover_name='actor', 
        size='size', size_max=25
    )
    
    # Add map background + contours on top or bottom axis
    fig = fig.add_trace(
        go.Surface(
            z=visual_data,
            colorbar_x=-0.1,
            contours_z=dict(
                show=True, usecolormap=True, 
                highlightcolor="limegreen", 
                project_z=True
            )
        )
    )

    # # Create an animated scatter plot
    # fig_scatter = px.scatter(actors_df, x='x', y='y',
    #     color='actor', animation_group='actor',
    #     animation_frame='epoch', hover_name='actor', 
    #     size='size', size_max=10
    # )

    # # Add the scatter data to the contour plot
    # fig.add_trace(fig_scatter.data[0], row=1, col=2)
    # fig.add_trace(fig_scatter.data[1], row=1, col=2)
    # fig.add_trace(fig_scatter.data[2], row=1, col=2)
    # fig.add_trace(fig_scatter.data[3], row=1, col=2)
    # fig.add_trace(fig_scatter.data[4], row=1, col=2)
    # update_layout = {
    #     'sliders': fig_scatter.layout['sliders'],
    #     'updatemenus': fig_scatter.layout['updatemenus']
    # }
    # fig.update_layout(update_layout)
    # fig.frames = fig_scatter.frames

    # # Draw the contour plot using the real dataframe
    # fig.add_trace(go.Contour(z = visual_data), row=1, col=2)

    # # Workaround to have both working frames and the contour as a background
    # fig.data = fig.data[::-1]
    # for i in range(len(fig.frames)):
    #     wrkan = [{}]
    #     wrkan.extend(fig.frames[i].data)
    #     fig.frames[i].data = wrkan
    
    # Finally, display the whole figure
    fig = fig.update_layout(title=title, autosize=False,
                            width=width, height=height,
                            margin=margin, 
                            scene={'aspectmode': 'cube',
                            'xaxis': {'range': [0, VISUAL_W],
                                    'rangemode': 'nonnegative',
                                    'autorange': False},
                            'yaxis': {'range': [0, VISUAL_H],
                                    'rangemode': 'nonnegative',
                                    'autorange': False}},
                            yaxis={'range': [0, VISUAL_H],
                                    'rangemode': 'nonnegative',
                                    'autorange': False},
                            xaxis={'range': [0,VISUAL_W],
                                    'rangemode': 'nonnegative',
                                    'autorange': False},
                            scene_camera = dict(
                                up=dict(x=0, y=0, z=1),
                                center=dict(x=0, y=0, z=0),
                                eye=dict(x=1.25, y=-1.25, z=1.2)
                            )
    )
    fig.write_html('tmp.html', auto_open=True)
