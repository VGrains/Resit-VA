import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import copy
import joblib

pd.options.mode.chained_assignment = None  

def create_fire_map(df, mapbox_access_token):
    '''
    Creates the large bubble map indicating all the fires. 
    Requires a dataframe and a mapbox access token.
    '''
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
    '''
    Creates the small State Chloropleth map with the total amount of fires per state. 
    Requires a dataframe.
    '''
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
    '''
    Creates the small Polar plot indicating the number of fires per month, or the average size of the fires per month. 
    Requires a dataframe and a boolean to indicate which representation is wanted.
    '''
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
    '''
    Creates the small bar chart indicating frequencies per fire cause. 
    Requires a dataframe.
    '''
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
    '''
    Searches for similar fires based on the user input using cosine similarity. 
    !! First filters the dataframe on the user-inputted discovery_month and vegetation code !!
    This might be removed in the future if the results are too few too often.

    Also predicts the fire_size_class and putout_time based on the user input with around 67% accuracy.
    Loads a trained model stored in a picke file.
    Requires a dataframe and all the inputted form data from the dbc.Form.
    '''

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
    DTC = joblib.load('SVM2.joblib')
    predicted_class = DTC.predict(query_array_predict)

    # Load the Decision Tree Classifier and predict the fire size class
    NN = joblib.load('ANN_reg2.joblib')
    predicted_putout = NN.predict(query_array_predict)
    predicted_putout = round(float(predicted_putout[0]), 2)
    # Map the predicted class to its string representation
    class_mapping = {
        0:'B',
        1:'C',
        2:'D',
        3:'E',
        4:'F',
        5:'G'
        }
    
    correct_class = class_mapping[predicted_class[0]]

    return sorted_df, correct_class, predicted_putout

    
    
