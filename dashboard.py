from dash import Dash, dcc, Output, Input, html  # pip install dash
import dash_bootstrap_components as dbc    # pip install dash-bootstrap-components
import data
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd



app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


colors = {
    'background_element': '#516096',
    'text': '#fafafc'
    'background'
}

df = pd.read_csv('FW_Veg_Rem_Combined.csv')
df['disc_clean_date'] = pd.to_datetime(df['disc_clean_date'])
df.sort_values(by='disc_clean_date', inplace = True) 

# Normalize the fire_size column
min_fire_size = df['fire_size'].min()
max_fire_size = df['fire_size'].max()
df['fire_size_norm'] = (df['fire_size'] - min_fire_size) / (max_fire_size - min_fire_size)



# Ensure lat and long are passed as floats
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)



## Help
app.layout = html.Div([ 
    html.Div([
    html.H1('Article Analysis & Clustering', style={'padding':'20px', 'textAlign': 'center'})
    ]),

    dbc.Row([
        html.Iframe(id='heatmap', srcDoc=open('heatmap.html', 'r').read(), width='80%')
            # dcc.Graph(
            #     id='heatmap',
            #     config={'staticPlot': False},
            #     figure={
            #         'data': [],
            #         'layout': {
            #             'width': '100%',
            #             'height': '80vh'
            #         }
            #     }
            # ) 
        ], id="test", style={'margin':'auto'}, align='center'),

    # dbc.Row([
    # # Add the user input box centered in the middle
    #     html.Div([
    #         html.H2('Enter your keyword:', style={'padding':'20px', 'border': '1px solid smokegray'}),
    #         dcc.Input(id='input-box', 
    #                 type='text', 
    #                 value=''
    #         ),
    #     ], style={'textAlign': 'center'}),
    # ]),
    dbc.Row([
        dbc.Col([
            # html.Div([
            #     html.H4("Classified cluster: ", id='cluster_classification', style={'border': '1px solid smokegray', 'text-align': 'center'}),
            #     html.H4("Cluster top keywords: ", id='cluster_classification_keywords', style={'border': '1px solid smokegray', 'text-align': 'center'})
            # ]),
            dcc.Graph(
                id="polar", 
                figure={}, 
                style={'width':'100%', 'margin':25, 'border': '1px solid smokegray'}
            )
        ], width=4),
        dbc.Col([
            
            dcc.Graph(
                id="state_map", 
                figure={}, 
                style={'width':'100%', 'margin':25, 'border': '1px solid smokegray'}
            )
        ], width=8)
    ], align='center', justify='center'),
])

@app.callback(
    Output("heatmap", component_property='srcDoc'),
    Input("test", component_property="children")
)

def update_map(children):
    map_html = data.create_heatmap_with_circles(df)
    return map_html

@app.callback(
    Output("state_map", component_property='figure'),
    Input("test", component_property="children")
)

def update_state_map(children):
    fig = data.create_state_map(df)
    return fig

@app.callback(
    Output("polar", component_property='figure'),
    Input("test", component_property="children"),
    Input("state_map", component_property='hoverData')
)

def update_polar_plot(children, hov_data):
    if hov_data is None:
        fig = data.create_polar_plot(df)
    else:
        print(hov_data)
        hov_location = hov_data['points'][0]['location']
        df_hov = df[df.state == hov_location]
        fig = data.create_polar_plot(df_hov)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)