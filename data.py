import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler,LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import folium
from folium.plugins import HeatMap, HeatMapWithTime
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
import copy
import joblib

pd.options.mode.chained_assignment = None  # default='warn'


# def create_fire_map(df, mapbox_access_token):

#     # Define a dictionary to map fire_size_class values to specific colors
#     color_mapping = {
#         'B': '#F4BE06',
#         'C': '#FAA307',
#         'D': '#E85D04',
#         'E': '#D00000',
#         'F': '#9D0208',
#         'G': '#43030b',
#         # Add more mappings as needed based on your data
#     }

#     size_mapping = {
#         'B': 1,
#         'C': 2,
#         'D': 4,
#         'E': 8,
#         'F': 16,
#         'G': 32,
#         # Add more mappings as needed based on your data
#     }

#     marker_colors = df["fire_size_class"].map(color_mapping)
#     marker_sizes = df["fire_size_class"].map(size_mapping)

#     # Create the scattermapbox plot
#     fig = px.scatter_mapbox(df,
#                             lat="latitude",
#                             lon="longitude",
#                             size=marker_sizes,  # Use the marker_sizes for size of the markers
#                             color=df['fire_size_class'],  # Use the marker_colors for color of the markers
#                             color_discrete_map=color_mapping,  # Map the color names to colors
#                             hover_name="fire_name",
#                             hover_data={"fire_size": True},
#                             zoom=4,
#                             height=800,
#                             )

#     # Set the mapbox properties
#     fig.update_layout(
#         mapbox=dict(
#             accesstoken=mapbox_access_token,
#             center=dict(lat=39.8283, lon=-98.5795),
#             style='dark',  # Set the URL of your custom Mapbox style JSON file here
#         ),
#         margin=dict(r=0, t=0, l=0, b=0),
#         showlegend=True,
#         legend=dict(
#             bgcolor='#242424',  # Set the background color of the legend
#             font=dict(color='lightgrey'),  # Set the font color of the legend text
#             title=dict(text='Fire Size Classes', font=dict(color='white')),  # Set the title of the legend with a custom color
#             orientation='v',  # Set the legend orientation to vertical
#             # x=1.05,  # Set the x-position of the legend (0.0 to 1.0)
#             y=0.5,   # Set the y-position of the legend (0.0 to 1.0)
#             yanchor='middle',  # Anchor the legend vertically at the middle
#             xanchor='left'     # Anchor the legend horizontally to the left
#         ),
#         paper_bgcolor='#242424',  # Set the color of the whitespace
#     )

#     return fig

def create_fire_map(df, mapbox_access_token):
    
    
    # Define a dictionary to map fire_size_class values to specific colors
    color_mapping = {
        'B': '#FAA307',
        'C': '#F4BE06',
        'D': '#E85D04',
        'E': '#D00000',
        'F': '#9D0208',
        'G': '#43030b',
        # Add more mappings as needed based on your data
    }
     
    size_mapping = {
        'B': 1,
        'C': 2,
        'D': 4,
        'E': 8,
        'F': 16,
        'G': 32,
        # Add more mappings as needed based on your data
    }

    marker_colors = df["fire_size_class"].map(color_mapping)
    marker_sizes = df["fire_size_class"].map(size_mapping)

    fig = go.Figure()

    # Loop through unique fire_size_class values
    # for fire_class in df["fire_size_class"].unique():
    #     filtered_df = df[df["fire_size_class"] == fire_class]

    #     marker_colors = filtered_df["fire_size_class"].map(color_mapping)
    #     marker_sizes = filtered_df["fire_size_class"].map(size_mapping)

    main_trace = go.Scattermapbox(
        lat=df["latitude"],
        lon=df["longitude"],
        mode="markers",
        marker=dict(size=marker_sizes, color=marker_colors, sizemode="diameter"),
        unselected={'marker': {'opacity': 0.01}},
        selected={'marker': {'opacity': 1, 'size': 25}},
        hovertext=df["fire_name"],
        hoverinfo="text",
        text=df["fire_size"],
        hoverlabel=dict(namelength=0),
        name='Full dataset'
    )

    fig.add_trace(main_trace)

    # Create dummy traces for each fire size class to display in the legend
    # for fire_class in df["fire_size_class"].unique():
    #     filtered_df = df[df["fire_size_class"] == fire_class]

    #     marker_colors = filtered_df["fire_size_class"].map(color_mapping)
    #     marker_sizes = filtered_df["fire_size_class"].map(size_mapping)

    for fire_class in df["fire_size_class"].unique():

        trace = go.Scattermapbox(
            lat=[0], lon=[0],  # Empty data for the dummy trace
            marker=dict(size=0, color=color_mapping[fire_class], opacity=0.7),
            mode="markers",
            name=f'Fire Size Class {fire_class}',
        )
         # Set the name of the trace for legend

        fig.add_trace(trace)
   

    # Create the layout for the map
    fig.update_layout(
        uirevision='foo',
        clickmode='event+select',
        mapbox=dict(
            accesstoken=mapbox_access_token,  # Set the Mapbox access token here
            center=dict(lat=39.8283, lon=-98.5795),
            zoom=4,
            style='dark',  # Set the URL of your custom Mapbox style JSON file here
        ),
        height=800,
        margin=dict(r=0, t=0, l=0, b=0),
        showlegend=True,  # Show the legend
        legend=dict(
            bgcolor='#242424',  # Set the background color of the legend
            font=dict(color='lightgrey'),  # Set the font color of the legend text
            title=dict(text='Fire Size Classes', font=dict(color='white')),  # Set the title of the legend with a custom color
            y=0.5, 
            itemsizing='constant',  # Set the size of the indicators to a constant value
            itemwidth=30,  # Set the width of the indicators in pixels
        ), 
        paper_bgcolor='#242424'
    )
    
    return fig

def create_state_map(df):

    df4 = copy.copy(df[['state', 'fire_size']])

    df4 = df4.groupby('state').count().reset_index()
    fig = go.Figure(data=go.Choropleth(
        locations=df4['state'], # Spatial coordinates
        z = df4['fire_size'].astype(float), # Data to be color-coded
        locationmode = 'USA-states', # set of locations match entries in `locations`
        colorscale = 'Reds',
        colorbar_title = "Amount of forest fires",
    ))

    fig.update_layout(
        title_text = 'Forest fires per state',
        geo=dict(
            scope='usa',  # limit map scope to USA
            bgcolor='#242424',  # Set the background color of the map area
            showframe=False,
            showland=False,
            showcountries=True
        ),
        paper_bgcolor='#242424',
        font=dict(color='lightgrey')
    )


    return fig

def create_polar_plot(df, avg_fire_size=False):
    # Define a mapping to convert month names to numeric representation
    month_dict = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }

    df1 = copy.copy(df[['discovery_month', 'fire_size']])

    if avg_fire_size == False:
        df2 = df1.groupby('discovery_month').count().reset_index()
    else:
        df2 = df1.groupby('discovery_month').mean().reset_index()

    custom_month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Sort the DataFrame by the 'Month' column using the custom order
    # df2['discovery_month'] = pd.Categorical(df2['discovery_month'], categories=custom_month_order, ordered=True)
    df2.sort_values(by='discovery_month', inplace=True)
    df2['discovery_month'] = df2['discovery_month'].map(month_dict)
    
    # # # df2['state'] = df2['state'].astype(int)
    fig = px.line_polar(df2, r=df2['fire_size'], theta=df2['discovery_month'],line_close=True)
    fig.update_layout(
    paper_bgcolor='#242424',
    polar=dict(
        bgcolor='#474747',  # Set the background color of the polar plot area
    ),
    font=dict(color='lightgrey')
    )
    fig.update_traces(line_color='#cc4e5b', line_width=5)
    
    return fig

def create_bar_plot(df):
    df5 = copy.copy(df[['stat_cause_descr', 'fire_size']])

    df5 = df5.groupby('stat_cause_descr').count().reset_index()

    fig = px.bar(df5, x='stat_cause_descr', y='fire_size', text_auto='.2s')

    fig.update_layout(
        paper_bgcolor='#242424',
        plot_bgcolor='#242424',  # Set the background color of the polar plot area
        font=dict(color='lightgrey'),
        yaxis=dict(gridcolor='#474747'), # Set the color of the grid lines along the x-axis
        )

    fig.update_traces(marker_color='#cc4e5b', marker=dict(line=dict(width=0)))

    return fig

def search_similar_fires(df, latitude, longitude, discovery_month, vegetation, temperature, wind, humidity):

    query_array = np.array([latitude, longitude, temperature, wind, humidity]).reshape(1,-1)
    query_array_predict = np.array([discovery_month, vegetation, latitude, longitude, temperature, wind, humidity]).reshape(1, -1)

    df_filtered = df[(df['discovery_month']==discovery_month) & (df['Vegetation']==vegetation)]
    # df_filtered_withclass = df_filtered[['ID', 'latitude', 'longitude',  'fire_size_class', 'Temp_pre_7', 'Wind_pre_7', 'Hum_pre_7']]
    df_onlynumeric = df_filtered[['latitude', 'longitude', 'Temp_pre_7', 'Wind_pre_7', 'Hum_pre_7']]
    
    data_array = df_onlynumeric.values

    # Compute the cosine similarity between the query vector and all article vectors
    similarity_scores = cosine_similarity(data_array, query_array)

    # Get the indices of the most similar articles
    most_similar_indices = similarity_scores.argsort(axis=0)[::-1][:10]
    most_similar_indices = [x[0] for x in most_similar_indices]
    sim = [similarity_scores[i][0] for i in most_similar_indices]
    
    sorted_df = df_filtered.iloc[most_similar_indices, :]
    sorted_df['similarity_score'] = sim
    sorted_df.sort_values(by=['fire_size_class','similarity_score'], inplace=True,
               ascending = [False, True])

    SVC = joblib.load('DC.joblib')
    
    predicted_class = SVC.predict(query_array_predict)

    return sorted_df, predicted_class

# def get_info_on_click(df, latitude, longitude):
    
    
