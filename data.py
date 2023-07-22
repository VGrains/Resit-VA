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

pd.options.mode.chained_assignment = None  

def create_fire_map(df, mapbox_access_token):
    
    # Dictionary to map fire_size_class values to specific colors
    color_mapping = {
        'B': '#FAA307',
        'C': '#F4BE06',
        'D': '#E85D04',
        'E': '#D00000',
        'F': '#9D0208',
        'G': '#43030b',
        }
     
    # Dictionary to map fire_size_class values to specific marker sizes
    size_mapping = {
        'B': 1,
        'C': 2,
        'D': 4,
        'E': 8,
        'F': 16,
        'G': 32,
    }

    marker_colors = df["fire_size_class"].map(color_mapping)
    marker_sizes = df["fire_size_class"].map(size_mapping)

    # Create the main trace
    fig = go.Figure()
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
    for fire_class in df["fire_size_class"].unique():
        trace = go.Scattermapbox(
            lat=[0], lon=[0],  # Empty data for the dummy trace
            marker=dict(size=0, color=color_mapping[fire_class], opacity=0.7),
            mode="markers",
            name=f'Fire Size Class {fire_class}',
        )

        fig.add_trace(trace)
   

    # Create the layout for the map
    fig.update_layout(
        uirevision='foo',
        clickmode='event+select',
        mapbox=dict(
            accesstoken=mapbox_access_token, 
            center=dict(lat=39.8283, lon=-98.5795),
            zoom=4,
            style='dark',  
        ),
        height=800,
        margin=dict(r=0, t=0, l=0, b=0),
        showlegend=True,  
        legend=dict(
            bgcolor='#242424',  
            font=dict(color='lightgrey'),  
            title=dict(text='Fire Size Classes', font=dict(color='white')),
            y=0.5, 
            itemsizing='constant',  
            itemwidth=30, 
        ), 
        paper_bgcolor='#242424'
    )
    
    return fig

def create_state_map(df):

    # Use a copy of the main dataframe
    dff = copy.copy(df[['state', 'fire_size']])

    # Count the occurrences of fires per state
    dff = dff.groupby('state').count().reset_index()

    # Create map
    fig = go.Figure(data=go.Choropleth(
        locations = dff['state'], # Spatial coordinates
        z = dff['fire_size'].astype(float), # Data to be color-coded
        locationmode = 'USA-states', # set of locations match entries in `locations`
        colorscale = 'Reds',
        colorbar_title = "Amount of forest fires",
    ))

    # Create layout
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
    
    # Dictionary to map month names to numeric representation
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

    # Use a copy of the main dataframe
    df1 = copy.copy(df[['discovery_month', 'fire_size']])

    # If user wants to display the average fire size, update the plot accordingly
    if avg_fire_size == False:
        df2 = df1.groupby('discovery_month').count().reset_index()
    else:
        df2 = df1.groupby('discovery_month').mean().reset_index()

    # Sort the values on month to get an accurate timeline
    df2.sort_values(by='discovery_month', inplace=True)
    df2['discovery_month'] = df2['discovery_month'].map(month_dict)

    # Create Polar plot
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

    # Use a copy of the main dataframe
    df1 = copy.copy(df[['stat_cause_descr', 'fire_size']])

    # Group the data and make the plot
    df1 = df1.groupby('stat_cause_descr').count().reset_index()
    fig = px.bar(df1, x='stat_cause_descr', y='fire_size', text_auto='.2s')
    fig.update_layout(
        paper_bgcolor='#242424',
        plot_bgcolor='#242424',  # Set the background color of the polar plot area
        font=dict(color='lightgrey'),
        yaxis=dict(gridcolor='#474747'), # Set the color of the grid lines along the x-axis
        )

    fig.update_traces(marker_color='#cc4e5b', marker=dict(line=dict(width=0)))

    return fig

def search_similar_fires(df, latitude, longitude, discovery_month, vegetation, temperature, wind, humidity):

    # Define a user query array for calculating cosine similarity, and a query_array_predict for the prediction task
    query_array = np.array([latitude, longitude, temperature, wind, humidity]).reshape(1,-1)
    query_array_predict = np.array([discovery_month, vegetation, latitude, longitude, temperature, wind, humidity]).reshape(1, -1)

    # Filter the dataframe to have only entries within the same month and with the same vegetation
    df_filtered = df[(df['discovery_month']==discovery_month) & (df['Vegetation']==vegetation)]
    df_onlynumeric = df_filtered[['latitude', 'longitude', 'Temp_pre_7', 'Wind_pre_7', 'Hum_pre_7']]
    
    # Compute the cosine similarity between the query array and all other rows
    data_array = df_onlynumeric.values
    similarity_scores = cosine_similarity(data_array, query_array)

    # Get the indices of the most similar articles
    most_similar_indices = similarity_scores.argsort(axis=0)[::-1][:10]
    most_similar_indices = [x[0] for x in most_similar_indices]
    sim = [similarity_scores[i][0] for i in most_similar_indices]
    
    # Create a dataframe using the indexes of the most similar articles
    sorted_df = df_filtered.iloc[most_similar_indices, :]
    sorted_df['similarity_score'] = sim
    sorted_df.sort_values(by=['fire_size_class','similarity_score'], inplace=True,
               ascending = [False, True])

    # Load the Decision Tree Classifier and predict the fire size class
    DTC = joblib.load('DC.joblib')
    predicted_class = DTC.predict(query_array_predict)

    # Map the predicted class to its string representation
    class_mapping = {
        1:'B',
        2:'C',
        3:'D',
        4:'E',
        5:'F',
        6:'G'
        }

    return sorted_df, predicted_class

    
    
