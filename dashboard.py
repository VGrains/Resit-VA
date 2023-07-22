from dash import Dash, dcc, Output, Input, State, html, dash_table
from dash.exceptions import PreventUpdate# pip install dash
import dash_bootstrap_components as dbc    # pip install dash-bootstrap-components
import data
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import dash_leaflet as dl
import warnings
import copy



app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])

# Basic app colors 
colors = {
    'background': '#474747',
    'cards': '#242424',
    'red': '#cc4e5b'
}

# Access token for the large map
mapbox_access_token = 'pk.eyJ1IjoiZnJpZWRyaWNlIiwiYSI6ImNsazl6M3NobTAyeXkzbHM1MzkzMHY0MHYifQ.qNFa2-YSY3sYOhux8EqaQg'

# Preprocessing the data
df = pd.read_csv('FW_Veg_Rem_Combined.csv')
df['disc_clean_date'] = pd.to_datetime(df['disc_clean_date'])
df = df.rename(columns={'Unnamed: 0': 'ID'})
df['fire_name'] = df['fire_name'].fillna(value='UNNAMED')
df['discovery_month'] = df.disc_clean_date.dt.month

# # Normalize the fire_size column
# min_fire_size = df['fire_size'].min()
# max_fire_size = df['fire_size'].max()
# df['fire_size_norm'] = (df['fire_size'] - min_fire_size) / (max_fire_size - min_fire_size)

# Ensure lat and long are passed as floats
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)

# Sort values on Fire Size Class
df.sort_values('fire_size_class', inplace=True)
df = df.reset_index()

# Size mapping for the markers
size_mapping = {
        'B': 1,
        'C': 2,
        'D': 4,
        'E': 8,
        'F': 16,
        'G': 32,
    }

df['size_marker'] = df["fire_size_class"].map(size_mapping)

# Years for the range slider
YEARS = [x for x in range(1992, 2016)] 

##################################################################################################################
##############################   CARDS FOR RANGESLIDER & LARGE MAP ################################################
##################################################################################################################


card_slider = dbc.Card([
    dbc.CardBody([
        dcc.RangeSlider(
            1992,
            2015,
            1,
            value=[1998, 2008], 
            marks={i: '{}'.format(i) for i in range(1992,2016,1)},
            id='years_slider',

    ),
    ])
], color='#242424', class_name='m-4')

card_largemap = dbc.Card([
    dbc.CardBody([
        dcc.Graph(
                id="large_map", 
                figure={}
            )
    ])
], color='#242424', class_name='m-4')


##################################################################################################################
##############################   FIRE INFO CARD & PREDICTION CARD ################################################
##################################################################################################################

card_info = dbc.Card([
    dbc.CardHeader(
        f'Chosen fire: ', id='clickheader', style={'color':'white'}
        ),
    dbc.CardBody([

    ], id='clickinfo')
], color='#242424', class_name='m-4', style={'height':'400px'})

card_extra = dbc.Card([
    dbc.CardHeader(
        "Most similar fires", style={'color':'lightgrey'}
    ),
    dbc.CardBody([
        html.Div([
            dash_table.DataTable(
            columns=[
                {'name': 'Fire Name', 'id': 'fire_name'},
                {'name': 'Fire Size', 'id': 'fire_size'},
                {'name': 'Fire Size Class', 'id': 'fire_size_class'},
                {'name': 'Discovery Date', 'id': 'disc_clean_date'}
            ],
            row_selectable='multi',
            selected_row_ids=[],
            selected_rows=[],
            id='datatable_fires'
    )], id='model_results'), 
    html.H2('Predicted Fire Size Class: No prediction', style={'color':'#cc4e5b'}, className='m-4', id='class_predict'),
    html.H2('Predicted Fire Putout time: No prediction', style={'color':'#cc4e5b'}, className='m-4', id='putout_time')
    ])
], color='#242424', class_name='m-4', style={'height':'450px'})

##################################################################################################################
##############################   CARDS FOR USER INPUT ###########################################################
##################################################################################################################


card_lat = dbc.Card([
    dbc.CardHeader([
        'Latitude of the fire'   
    ], style={'color':'#cc4e5b'}),
        dcc.Input(
            id="input_lat",
            type='number',
            value=19,
            placeholder="Latitude of the fire"
        )
], color='#242424', class_name='m-4')

card_lon = dbc.Card([
    dbc.CardHeader([
        'Longitude of the fire'   
    ], style={'color':'#cc4e5b'}),
        dcc.Input(
            id="input_lon",
            type='number',
            value=-67,
            placeholder="Longitude of the fire"
        )
], color='#242424', class_name='m-4')

card_discm = dbc.Card([
    dbc.CardHeader([
        'Discovery month as number'   
    ], style={'color':'#cc4e5b'}),
        dcc.Input(
            id="input_discm",
            type='number',
            value=7,
            placeholder="Discovery month in numbers (eg. Jan = 1)"
        )
], color='#242424', class_name='m-4')

card_veg = dbc.Card([
    dbc.CardHeader([
        'Vegetation code'   
    ], style={'color':'#cc4e5b'}),
        dcc.Input(
            id="input_veg",
            type='number',
            value=12,
            placeholder="Vegetation code"
        )
], color='#242424', class_name='m-4')

card_temp = dbc.Card([
    dbc.CardHeader([
        'Temperature last 7 days (avg)'   
    ], style={'color':'#cc4e5b'}),
        dcc.Input(
            id="input_temp",
            type='number',
            value=3.3,
            placeholder="Temperature"
        )
], color='#242424', class_name='m-4')

card_wind = dbc.Card([
    dbc.CardHeader([
        'Windspeed last 7 days (avg)'   
    ], style={'color':'#cc4e5b'}),
        dcc.Input(
            id="input_wind",
            type='number',
            value=3.3,
            placeholder="Windspeed"
        )
], color='#242424', class_name='m-4')

card_hum = dbc.Card([
    dbc.CardHeader([
        'Humidity last 7 days (avg)'   
    ], style={'color':'#cc4e5b'}),
        dcc.Input(
            id="input_hum",
            type='number',
            value=80,
            placeholder="Humidity"
        )
], color='#242424', class_name='m-4')

##################################################################################################################
###########################   CARDS ON BOTTOM ROW. POLARCHART, STATE MAP AND BARCHART ############################
##################################################################################################################

card_polar = dbc.Card([
    dbc.CardHeader([
        dcc.Dropdown(
                ['Total fires', 'Avg fire size'],
                'Total fires',
                id='drop', 
                clearable=False
            )
    ]),
    dbc.CardBody([
        dcc.Graph(
                id="polar", 
                figure={} 
            )
    ])
], color='#242424', class_name='m-4')

card_state = dbc.Card([
    dbc.CardHeader([
        html.H4(
            
            "Total forest fires per state" , className='m-1', style={'font-family': 'Georgia, serif', 'color': '#cc4e5b'}
        )
    ]),
    dbc.CardBody([
        dcc.Graph(
                id="state_map", 
                figure={}, 
            )
    ])
], color='#242424', class_name='m-4')

card_barchart = dbc.Card([
    dbc.CardHeader([
        html.H4(
            "Top Causes of fires" , className='m-1', style={'font-family': 'Georgia, serif', 'color': '#cc4e5b'}
        )
    ]),
    dbc.CardBody([
        dcc.Graph(
                id="bar", 
                figure={} 
            )
    ])
], color='#242424', class_name='m-4')

##################################################################################################################
#########################################   APP LAYOUT   #########################################################
##################################################################################################################

app.layout = html.Div([ 
    html.Div([
    # Large header
    html.H1('Forest Fire Analysis', style={'padding':'20px', 'textAlign': 'center', 'font-family': 'Georgia, serif', 'color': '#cc4e5b'})
    ]),

    # First row containing the rangeslider and large_map card, as well as the card with the fire info and datatable card 
    # in a seperate column.
    dbc.Row([
        dbc.Col([
            card_slider,
            card_largemap
            ], width=8),
        dbc.Col([
            card_info,
            card_extra
            ], width=4),
        ], id="row1", style={'margin':'auto'}, align='center', justify='center'),

    # Second row containing the cards for getting the user input
    dbc.Form(
        dbc.Row([
            dbc.Col(card_lat),
            dbc.Col(card_lon),
            dbc.Col(card_discm),
            dbc.Col(card_veg),
            dbc.Col(card_temp),
            dbc.Col(card_wind),
            dbc.Col(card_hum),
            dbc.Col(dbc.Button("Submit", color="danger", id='submit_fire_info', className="me-1"), width='auto'),
            html.Div(id='output-div')
        ], id="row2", style={'margin':'auto'}, align='center', justify='center')),

    # Third row containing the three plots at the bottom.
    # The polar plot, state map, and bar chart.
    dbc.Row([
        dbc.Col([
            card_polar
        ], width=4),
        dbc.Col([ 
            card_state
        ], width=4),
        dbc.Col([ 
            card_barchart
        ], width=4)
    ], align='center', justify='center'),
], id='row3', style={'backgroundColor': '#474747'})

##################################################################################################################
##############################   THE CALLBACK     #############################################################
##################################################################################################################


@app.callback(
    Output("large_map", component_property="figure"),
    Input("years_slider", component_property="value"),
)

def update_map(value):
    '''
    Updates the large map based on the rangeslider input.
    '''

    filtered_data = df[(df["disc_clean_date"].dt.year >= value[0]) & (df["disc_clean_date"].dt.year <= value[1])]
    fig = data.create_fire_map(filtered_data, mapbox_access_token)

    return fig

@app.callback(
    Output("clickinfo", component_property='children'),
    Input("large_map", component_property="clickData")
)

def update_clickinfo(clickData):
    '''
    Updates the information list that is seen on the card to the right of the map
    Based on what fire the user clicked on on the map.
    '''

    if clickData == None:
        return ''

    else:
        # Search for the corresponding data entry in the dataframe
        latitude = clickData['points'][0]['lat']
        longitude = clickData['points'][0]['lon']
        clicked_fire = df[(df['latitude']==latitude) & (df['longitude']==longitude)]

        # Color mapping dict used for coloring the list according to the Fire Size Class
        color_mapping = {
            'B': '#FAA307',
            'C': '#F4BE06',
            'D': '#E85D04',
            'E': '#D00000',
            'F': '#9D0208',
            'G': '#43030b',
            }
        
        # Determine color
        color_listgroup = color_mapping[clicked_fire['fire_size_class'].values[0]]

        # Change the font color to lightgrey if the background is too dark
        if color_listgroup == '#43030b' or color_listgroup == '#9D0208':
            selected_info_list = dbc.ListGroup(
            [
                dbc.ListGroupItem(f'Fire ID: {clicked_fire["ID"].values[0]}', color=color_listgroup, style={'color':'lightgrey'}),
                dbc.ListGroupItem(f'Date: {clicked_fire["disc_clean_date"].values[0]}', color=color_listgroup, style={'color':'lightgrey'}),
                dbc.ListGroupItem(f'State: {clicked_fire["state"].values[0]}', color=color_listgroup, style={'color':'lightgrey'}),
                dbc.ListGroupItem(f'Fire size: {clicked_fire["fire_size"].values[0]} acres', color=color_listgroup, style={'color':'lightgrey'}),
                dbc.ListGroupItem(f'Time until put out: {clicked_fire["putout_time"].values[0]}', color=color_listgroup, style={'color':'lightgrey'}),
                dbc.ListGroupItem(f'Vegetation: {clicked_fire["Vegetation"].values[0]}', color=color_listgroup, style={'color':'lightgrey'}),
                dbc.ListGroupItem(f'Cause: {clicked_fire["stat_cause_descr"].values[0]}', color=color_listgroup, style={'color':'lightgrey'}),

            ], flush=False),
            
        else:
            selected_info_list = dbc.ListGroup(
            [
                dbc.ListGroupItem(f'Fire ID: {clicked_fire["ID"].values[0]}', color=color_listgroup),
                dbc.ListGroupItem(f'Date: {clicked_fire["disc_clean_date"].values[0]}', color=color_listgroup),
                dbc.ListGroupItem(f'State: {clicked_fire["state"].values[0]}', color=color_listgroup),
                dbc.ListGroupItem(f'Fire size: {clicked_fire["fire_size"].values[0]} acres', color=color_listgroup),
                dbc.ListGroupItem(f'Time until put out: {clicked_fire["putout_time"].values[0]}', color=color_listgroup),
                dbc.ListGroupItem(f'Vegetation: {clicked_fire["Vegetation"].values[0]}', color=color_listgroup),
                dbc.ListGroupItem(f'Cause: {clicked_fire["stat_cause_descr"].values[0]}', color=color_listgroup),

            ], flush=False)
            
        return selected_info_list
    
@app.callback(
    Output("clickheader", component_property="children"),
    Input("large_map", component_property="clickData")
)

def update_clickedheader(clickData):
    '''
    Updates the header that is seen on the card to the right of the map
    Based on what fire the user clicked on on the map.
    '''

    if clickData == None:
        return 'No fire selected'
    else:
        latitude = clickData['points'][0]['lat']
        longitude = clickData['points'][0]['lon']
        clicked_fire = df[(df['latitude']==latitude) & (df['longitude']==longitude)]
        return f'{clicked_fire["fire_name"].values[0]}'


@app.callback(
    Output("state_map", component_property='figure'),
    Input("row1", component_property="children")
)

def update_state_map(children):
    '''
    Updates the State Map Plot.
    '''
    fig = data.create_state_map(df)
    return fig

@app.callback(
    Output("polar", component_property='figure'),
    Input("drop", component_property="value"),
    Input("state_map", component_property='hoverData')
)

def update_polar_plot(dropdown_value, hov_data):
    '''
    Creates a Polar plot of the total number of fires per month.
    If user hovers over the State Map, the Polar plot changes to show only data on that particular state.
    User can choose beween Total NR of Fires and Avg Number of Fires via a Dropdown.
    '''
    if dropdown_value == 'Total fires':
        if hov_data is None:
            fig = data.create_polar_plot(df)
        
        elif hov_data is not None:
            # Determine the state and filter the df
            hov_location = hov_data['points'][0]['location']
            df_hov = df[df.state == hov_location]
            fig = data.create_polar_plot(df_hov)

    elif dropdown_value == 'Avg fire size':
        if hov_data is None:
            fig = data.create_polar_plot(df, avg_fire_size=True)
        
        elif hov_data is not None:
            # Determine the state and filter the df
            hov_location = hov_data['points'][0]['location']
            df_hov = df[df.state == hov_location]
            fig = data.create_polar_plot(df_hov, avg_fire_size=True)
    
    return fig

@app.callback(
    Output("bar", component_property='figure'),
    Input("state_map", component_property='hoverData')
)

def update_bar_chart(hov_data):
    '''
    Creates a bar chart showing the most frequent causes of fires.
    If there is no hover data, it will show the total over the whole country.
    Otherwise shows information on a particular state.
    '''
    if hov_data is None:
        fig = data.create_bar_plot(df)
        
    elif hov_data is not None:
        # Determine the state and filter the df
        hov_location = hov_data['points'][0]['location']
        df_hov = df[df.state == hov_location]
        fig = data.create_bar_plot(df_hov)
    
    return fig

@app.callback(
    [Output('large_map', 'figure', allow_duplicate=True),
     Output("model_results", component_property="children"),
     Output("class_predict", component_property="children"),
     Output("putout_time", component_property="children")],
    [Input('submit_fire_info', 'n_clicks')],
    [State("input_lat", "value"),
    State("input_lon", "value"),
    State("input_discm", "value"),
    State("input_veg", "value"),
    State("input_temp", "value"),
    State("input_wind", "value"),
    State("input_hum", "value"),
    State('large_map', 'figure')],
    prevent_initial_call=True
)
def process_form_data(n_clicks, lat, long, discm, veg, temp, wind, hum, current_figure):
    '''
    Processes the user inputted data and searches for the most similar fires.
    Displays the most similar fires in a DataTable on the right of the large map, and shows them as Selected on the large map.
    Also predicts the Fire Size Class and Putout Time of the fire that the user inputted.
    '''

    if n_clicks is None:
        return current_figure, '', ''

    else: 
        # If no similar entries can be found, catch a ValueError that comes up and display a warning.
        try:
            # Get similar fires and the predicted class
            similar_fires, predicted_class, predicted_putouttime = data.search_similar_fires(df, lat, long, discm, veg, temp, wind, hum)
            
            top10 = []
            # Retrieve the matching entries in the main df 
            # (the other one can be filtered)
            for index, row in similar_fires.iterrows():
                matching_row_index = df[df['ID'] == row['ID']]
                top10.append(matching_row_index.index[0])
            
            # Get the top 5 most similar fires
            top5 = top10[:5]

            # Create new plot and highlight the top 5 similar fires on the large map.
            fig = data.create_fire_map(df, mapbox_access_token)
            fig.update_traces(selectedpoints=top5)

            # Create the datatable for the most similar fires
            df_datatable = df.loc[top5]
            df_datatable = df_datatable[['fire_name', 'fire_size', 'fire_size_class', 'disc_clean_date']]
            df_datatable['Index'] = df_datatable.index
            most_similar_fires = dash_table.DataTable(
                columns=[
                    {'name': 'Index', 'id':'index'},
                    {'name': 'Fire Name', 'id': 'fire_name'},
                    {'name': 'Fire Size', 'id': 'fire_size'},
                    {'name': 'Class', 'id': 'fire_size_class'},
                    {'name': 'Discovery Date', 'id': 'disc_clean_date'}
                ],
                data=df_datatable.to_dict('records'),
                row_selectable='multi',
                selected_row_ids=[],
                selected_rows=[0,1,2,3,4],
                id='datatable_fires'
            )

            # Format the predictions
            prediction_class = f'Predicted Fire Size Class: {predicted_class}'
            prediction_time = f'Predicted Fire Putout Time: {predicted_putouttime} days'

            return fig, most_similar_fires, prediction_class, prediction_time

        except ValueError as e:
            print(e)
            # Catch the ValueError and turn it into a warning
            alert = dbc.Alert(
                "No fires were found, try using other data.",
                id="alert-fade",
                dismissable=True,
                is_open=True,
                color="danger"
                ),
              
            return current_figure, alert, '', ''
    
@app.callback(
    Output("large_map", component_property='figure', allow_duplicate=True),
    [Input("datatable_fires", component_property='derived_virtual_data'), 
    Input("datatable_fires", component_property='selected_rows')],
    prevent_initial_call=True
)

def update_large_map_from_datatable(derived_virtual_data, selected_rows):
    '''
    Displays the selected points in the datatable on the large map
    '''
    if not selected_rows:
        raise PreventUpdate
    else:
        # Get the actual DataFrame indexes of the selected rows
        actual_indexes = [derived_virtual_data[index]['Index'] for index in selected_rows]

        # Create updated figure
        fig = data.create_fire_map(df, mapbox_access_token)
        fig.update_traces(selectedpoints=actual_indexes)
        return fig

if __name__ == '__main__':
    app.run_server(debug=True)