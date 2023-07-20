from dash import Dash, dcc, Output, Input, State, html  # pip install dash
import dash_bootstrap_components as dbc    # pip install dash-bootstrap-components
import data
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import dash_leaflet as dl



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
df.sort_values(by='disc_clean_date', inplace = True) 
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

card_lat = dbc.Card([
        dcc.Input(
            id="input_lat",
            type='number',
            placeholder="Latitude of the fire"
        )
], color='#242424', class_name='m-4')

card_lon = dbc.Card([
        dcc.Input(
            id="input_lon",
            type='number',
            placeholder="Longitude of the fire"
        )
], color='#242424', class_name='m-4')

card_wind = dbc.Card([
        dcc.Input(
            id="input_wind",
            type='number',
            placeholder="Windspeed"
        )
], color='#242424', class_name='m-4')

card_hum = dbc.Card([
        dcc.Input(
            id="input_hum",
            type='number',
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
            ]),
        ], id="test", style={'margin':'auto'}, align='center', justify='center'),

    dbc.Form(
        dbc.Row([
            dbc.Col(card_lat),
            dbc.Col(card_lon),
            dbc.Col(card_wind),
            dbc.Col(card_hum),
            dbc.Col(dbc.Button("Submit", color="danger", id='submit_fire_info', className="me-1"), width='auto'),
            html.Div(id='output-div')
        ], id="test", style={'margin':'auto'}, align='center', justify='center')),

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
], style={'backgroundColor': '#474747'})

@app.callback(
    Output("large_map", component_property="figure"),
    Input("years_slider", component_property="value")
)

def update_map(value):

    # Filter the fire data based on the size threshold
    print(value)
    filtered_data = df[(df["disc_clean_date"].dt.year >= value[0]) & (df["disc_clean_date"].dt.year <= value[1])]
    
    fig = data.create_fire_map(filtered_data, mapbox_access_token, mapbox_style_url)
    return fig

@app.callback(
    Output("state_map", component_property='figure'),
    Input("test", component_property="children")
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
    Output('output-div', 'children'),
    Input('submit_fire_info', 'n_clicks'),
    [State("input_lat", "value"),
    State("input_lon", "value"),
    State("input_wind", "value"),
    State("input_hum", "value")]
)
def process_form_data(n_clicks, lat, long, wind, hum):
    if n_clicks is None:
        # Process the data (e.g., save it to a database, send an email, etc.)
        # In this example, we just display the submitted data in the output div
        return 'submitted' 
    else:
        print(n_clicks, lat, long, wind, hum)


if __name__ == '__main__':
    app.run_server(debug=True)