from dash import Dash, dcc, Output, Input, State, html, dash_table  # pip install dash
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
import copy



app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])


colors = {
    'background': '#474747',
    'cards': '#242424',
    'red': '#cc4e5b'
}

mapbox_access_token = 'pk.eyJ1IjoiZnJpZWRyaWNlIiwiYSI6ImNsazl6M3NobTAyeXkzbHM1MzkzMHY0MHYifQ.qNFa2-YSY3sYOhux8EqaQg'
mapbox_style_url = 'mapbox://styles/friedrice/clk9ys5wi00sa01pe9jo7cmmz'

df = pd.read_csv('FW_Veg_Rem_Combined.csv')
df['disc_clean_date'] = pd.to_datetime(df['disc_clean_date'])
# df.sort_values(by='disc_clean_date', inplace = True) 
df = df.rename(columns={'Unnamed: 0': 'ID'})
df['fire_name'] = df['fire_name'].fillna(value='UNNAMED')
df['discovery_month'] = df.disc_clean_date.dt.month

# Normalize the fire_size column
min_fire_size = df['fire_size'].min()
max_fire_size = df['fire_size'].max()
df['fire_size_norm'] = (df['fire_size'] - min_fire_size) / (max_fire_size - min_fire_size)



# Ensure lat and long are passed as floats
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)

df.sort_values('fire_size_class', inplace=True)
df = df.reset_index()

size_mapping = {
        'B': 1,
        'C': 2,
        'D': 4,
        'E': 8,
        'F': 16,
        'G': 32,
        # Add more mappings as needed based on your data
    }

df['size_marker'] = df["fire_size_class"].map(size_mapping)

YEARS = [x for x in range(1992, 2016)] 

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
                # style={'width':'100%', 'margin':25, 'border': '1px solid smokegray'}
            )
    ])
], color='#242424', class_name='m-4')

card_info = dbc.Card([
    dbc.CardHeader(f'Chosen fire: ', id='clickheader', style={'color':'white'}),
    dbc.CardBody([], id='clickinfo')
], color='#242424', class_name='m-4', style={'height':'400px'})

card_extra = dbc.Card([
    dbc.CardHeader([
        
    ]),
    dbc.CardBody([html.Div([
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
    html.H2('Predicted putout time: No prediction', style={'color':'#cc4e5b'}, className='m-4', id='putout_time')
    ])
], color='#242424', class_name='m-4', style={'height':'350px'})

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
        'Discovery month in numbers (Jan = 1)'   
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
        'Windspeed'   
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
        'Humidity'   
    ], style={'color':'#cc4e5b'}),
        dcc.Input(
            id="input_hum",
            type='number',
            value=80,
            placeholder="Humidity"
        )
], color='#242424', class_name='m-4')

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
                # style={'width':'100%', 'margin':25, 'border': '1px solid smokegray'}
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
                # style={'width':'100%', 'margin':25, 'border': '1px solid smokegray'}
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
                # style={'width':'100%', 'margin':25, 'border': '1px solid smokegray'}
            )
    ])
], color='#242424', class_name='m-4')



app.layout = html.Div([ 
    html.Div([
    html.H1('Forest Fire Analysis', style={'padding':'20px', 'textAlign': 'center', 'font-family': 'Georgia, serif', 'color': '#cc4e5b'})
    ]),

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

@app.callback(
    Output("large_map", component_property="figure"),
    [Input("years_slider", component_property="value"),
     Input("large_map", component_property="clickData")],
)

def update_map(value, clickData):
    
    # filtered_data = df[(df["disc_clean_date"].dt.year >= value[0]) & (df["disc_clean_date"].dt.year <= value[1])]
    fig = data.create_fire_map(df, mapbox_access_token)

    return fig

@app.callback(
    Output("clickinfo", component_property='children'),
    Input("large_map", component_property="clickData")
)

def update_clickinfo(clickData):

    # Filter the fire data based on the size threshold
    if clickData == None:
        return ''

    else:
        latitude = clickData['points'][0]['lat']
        longitude = clickData['points'][0]['lon']
        clicked_fire = df[(df['latitude']==latitude) & (df['longitude']==longitude)]

        color_mapping = {
            'B': '#FAA307',
            'C': '#F4BE06',
            'D': '#E85D04',
            'E': '#D00000',
            'F': '#9D0208',
            'G': '#43030b',
            }
        
        color_listgroup = color_mapping[clicked_fire['fire_size_class'].values[0]]

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

    # Filter the fire data based on the size threshold
    if clickData == None:
        return 'No fire selected'
    else:
        latitude = clickData['points'][0]['lat']
        longitude = clickData['points'][0]['lon']
        print(clickData['points'])
        clicked_fire = df[(df['latitude']==latitude) & (df['longitude']==longitude)]
        return f'{clicked_fire["fire_name"].values[0]}'


@app.callback(
    Output("state_map", component_property='figure'),
    Input("row1", component_property="children")
)

def update_state_map(children):
    fig = data.create_state_map(df)
    return fig

@app.callback(
    Output("polar", component_property='figure'),
    Input("drop", component_property="value"),
    Input("state_map", component_property='hoverData')
)

def update_polar_plot(dropdown_value, hov_data):
    if dropdown_value == 'Total fires':
        if hov_data is None:
            fig = data.create_polar_plot(df)
        
        elif hov_data is not None:
            hov_location = hov_data['points'][0]['location']
            df_hov = df[df.state == hov_location]
            fig = data.create_polar_plot(df_hov)

    elif dropdown_value == 'Avg fire size':
        if hov_data is None:
            fig = data.create_polar_plot(df, avg_fire_size=True)
        
        elif hov_data is not None:
            hov_location = hov_data['points'][0]['location']
            df_hov = df[df.state == hov_location]
            fig = data.create_polar_plot(df_hov, avg_fire_size=True)
    
    return fig

@app.callback(
    Output("bar", component_property='figure'),
    Input("state_map", component_property='hoverData')
)

def update_state_map(hov_data):
    if hov_data is None:
        fig = data.create_bar_plot(df)
        
    elif hov_data is not None:
        hov_location = hov_data['points'][0]['location']
        df_hov = df[df.state == hov_location]
        fig = data.create_bar_plot(df_hov)
    
    return fig

@app.callback(
    [Output('large_map', 'figure', allow_duplicate=True),
     Output("model_results", component_property="children"),
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
    if n_clicks is None:
        
        return current_figure, '', ''

    else: 
        try:
            # Your code that may raise the ValueError
            similar_fires, predicted_class = data.search_similar_fires(df, lat, long, discm, veg, temp, wind, hum)
            top10 = []
            for index, row in similar_fires.iterrows():
                matching_row_index = df[df['ID'] == row['ID']]
                top10.append(matching_row_index.index[0])
            
            top5 = top10[:5]
            fig = data.create_fire_map(df, mapbox_access_token)
            
            df_datatable = df.loc[top5]
            lats = df_datatable.latitude.values
            longs = df_datatable.longitude.values

            # for trace in fig['data']:
            #     latitudes = trace['lat']
            #     longitudes = trace['lon']
            #     coordinates = zip(latitudes, longitudes)
            #     for i, j in coordinates:
            #         if i == lats
            #     point_indexes = list(range(len(latitudes)))
            #     all_point_indexes.extend(point_indexes)


            
            # # Make a deep copy of the current_figure to avoid modifying the original data
            
            print(df_datatable)
            # update_dict = {'B':[],
            #           'C':[] ,
            #           'D':[],
            #           'E':[],
            #           'F':[],
            #           'G':[]}
            
            # for index, row in df_datatable.iterrows():
            #     fire_class = row['fire_size_class']
            #     update_dict[fire_class].append(index)
            # print(update_dict)
            # for trace in current_figure['data']:
            #     trace['selectedpoints'] = top5

            # print(update_dict[trace['name']])
        
            fig.update_traces(selectedpoints=top5)
            
            df_datatable = df_datatable[['fire_name', 'fire_size', 'fire_size_class', 'disc_clean_date']]
            most_similar_fires = dash_table.DataTable(
                columns=[
                    {'name': 'Fire Name', 'id': 'fire_name'},
                    {'name': 'Fire Size', 'id': 'fire_size'},
                    {'name': 'Fire Size Class', 'id': 'fire_size_class'},
                    {'name': 'Discovery Date', 'id': 'disc_clean_date'}
                ],
                data=df_datatable.to_dict('records'),
                row_selectable='multi',
                selected_row_ids=[],
                selected_rows=[],
                id='datatable_fires'
            )


            

            return fig, most_similar_fires, predicted_class

        except ValueError as e:
            # Catch the ValueError and turn it into a warning
            warnings.warn(str(e))
            return None  # Retur
    

# @app.callback(
#     Output("putout_time", component_property='children'),
#     Input("datatable_fires", component_property='selected_row_ids'),
#     State('large_map', 'figure'), prevent_initial_call=True
# )

# def update_large_map_from_datatable(selected_rows, current_figure):
#     if selected_rows is None:
#         pass
 
#     else:
#         fig = go.Figure(current_figure)
#         fig.update_traces(selectedpoints=selected_rows)
        return fig
    

# @app.callback(
#     Output("putout_time", component_property='children', allow_duplicate=True),
#     Input("datatable_fires", component_property='selected_row_ids'),
#     State('large_map', 'figure'), prevent_initial_call=True
# )

# def update_large_map_from_datatable(selected_rows, current_figure):
#     if selected_rows is None:
#         pass
 
#     else:
#         fig = go.Figure(current_figure)
#         fig.update_traces(selectedpoints=selected_rows)
#         return fig

if __name__ == '__main__':
    app.run_server(debug=True)