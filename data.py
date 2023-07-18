import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler,LabelEncoder
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

def create_state_map(df):

    df4 = copy.copy(df[['state', 'fire_size']])

    df4 = df4.groupby('state').count().reset_index()
    df4
    fig = go.Figure(data=go.Choropleth(
        locations=df4['state'], # Spatial coordinates
        z = df4['fire_size'].astype(float), # Data to be color-coded
        locationmode = 'USA-states', # set of locations match entries in `locations`
        colorscale = 'Reds',
        colorbar_title = "Amount of forest fires",
    ))

    fig.update_layout(
        title_text = 'Forest fires per state',
        geo_scope='usa', # limite map scope to USA
    )

    return fig

def create_polar_plot(df):
    # Define a mapping to convert month names to numeric representation
    month_to_number = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    df1 = copy.copy(df[['discovery_month', 'state']])

    df2 = df1.groupby('discovery_month').count().reset_index()

    custom_month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Sort the DataFrame by the 'Month' column using the custom order
    df2['discovery_month'] = pd.Categorical(df2['discovery_month'], categories=custom_month_order, ordered=True)
    df2.sort_values(by='discovery_month', inplace=True)
    df2

    # # # df2['state'] = df2['state'].astype(int)
    fig = px.line_polar(df2, r=df2['state'], theta=df2['discovery_month'],line_close=True,template="plotly_dark",)
    
    return fig