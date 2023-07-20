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



def create_heatmap_with_circles(df):

    # Create map object
    map_obj = folium.Map(location = [38.27312, -98.5821872], zoom_start = 4)


    # Only show fires with magnitude greater or equal to 10, otherwise it gets overwhelming
    df_onlylarge = df[df['fire_mag'] >= 10.0]
    df_heat = df_onlylarge[['longitude','latitude', 'fire_size_norm', 'disc_clean_date']]

    # List comprehension to make out list of lists
    heat_data = [[row['latitude'],row['longitude'],row['fire_size_norm']] for index, row in df_heat.iterrows()]
    heat_time_dates =  [[row['disc_clean_date']] for index, row in df_heat.iterrows()]

    # HeatMapWithTime(heat_data, index=heat_time_dates, radius=10, blur=10).add_to(map_obj)
    HeatMap(heat_data, radius=10, blur=10).add_to(map_obj)

    for i in range(0, len(df_onlylarge)):
        radius = math.sqrt((df_onlylarge.iloc[i]['fire_size']*4046.856422)/math.pi)
        folium.Circle(
        location = [df_onlylarge.iloc[i]['latitude'], df_onlylarge.iloc[i]['longitude']],
        radius = radius,
        tooltip = '<li><bold> Fire size: ' + str(df_onlylarge.iloc[i]['fire_size']) + ' acres' + 
                '<li><bold> Fire size class: ' + str(df_onlylarge.iloc[i]['fire_size_class']) +
                '<li><bold> Date: ' + str(df_onlylarge.iloc[i]['disc_clean_date'])
                
        ).add_to(map_obj)

    map_obj.save('heatmap.html')

def create_fire_map(df, mapbox_access_token, mapbox_style_url):
    
    
    # Define a dictionary to map fire_size_class values to specific colors
    color_mapping = {
        'B': '#636EFA',
        'C': '#EF553B',
        'D': '#00CC96',
        'E': '#AB63FA',
        'F': '#FFA15A',
        'G': '#19D3F3',
        # Add more mappings as needed based on your data
    }

    marker_colors = df["fire_size_class"].map(color_mapping)

    # Create the scatter mapbox trace
    scattermapbox_trace = go.Scattermapbox(
        lat=df["latitude"],
        lon=df["longitude"],
        mode="markers",
        marker=dict(size=4, color=marker_colors, sizemode="diameter"),
        hovertext=df["fire_name"],
        hoverinfo="text",
        text=df["fire_size"],
        hoverlabel=dict(namelength=0),  # Hide hover label name
        # customdata=df["fire_size_class"],  # Use customdata to store fire_size_class for coloring
        # # marker_color=df["fire_size_class"],  # Color code based on fire size class variable
        # colorscale="Viridis",  # Choose a colorscale for coloring
        # colorbar_title="Fire Size Class",
    )

    # Create the layout for the map
    map_layout = go.Layout(
        uirevision='foo',
        mapbox=dict(
            accesstoken=mapbox_access_token,  # Set the Mapbox access token here
            center=dict(lat=39.8283, lon=-98.5795),
            zoom=4,
            style='dark',  # Set the URL of your custom Mapbox style JSON file here
        ),
        height=800,
        margin=dict(r=0, t=0, l=0, b=0),
        showlegend=False,
    )

    # Create the map figure
    fig = go.Figure(data=[scattermapbox_trace], layout=map_layout)
    
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
    month_to_number = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    df1 = copy.copy(df[['discovery_month', 'fire_size']])

    if avg_fire_size == False:
        df2 = df1.groupby('discovery_month').count().reset_index()
    else:
        df2 = df1.groupby('discovery_month').mean().reset_index()

    custom_month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Sort the DataFrame by the 'Month' column using the custom order
    df2['discovery_month'] = pd.Categorical(df2['discovery_month'], categories=custom_month_order, ordered=True)
    df2.sort_values(by='discovery_month', inplace=True)
    df2

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




def search_similar_fires(df, discovery_month, vegetation):

    query_array = np.array([19.0, -67.0, 3.3, 80]).reshape(1,-1)

    df_filtered = df[(df['discovery_month']==discovery_month) & (df['Vegetation']==vegetation)]
    df_filtered_withclass = df_filtered[['ID', 'latitude', 'longitude',  'fire_size_class', 'Wind_cont', 'Hum_cont']]
    df_onlynumeric = df_filtered[['latitude', 'longitude', 'Wind_cont', 'Hum_cont']]
    
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

    return sorted_df

# simfires = search_similar_fires(df, 7, 12)