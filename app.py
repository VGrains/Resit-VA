from dash import Dash, html, dcc, Input, Output
import dash
from jbi100_app.views.menu import generate_homepage_layout, generate_figure1_layout, generate_figure2_layout, generate_figure3_layout

app = Dash(__name__, use_pages=True)

app.layout = html.Div(children=[
    html.Div(id='global_header', children='JBI100 - Visualization project group 58'),
    dcc.Tabs(id="global_header_tabs", value="HomeTab", children=[
        dcc.Tab(label="Home", value="HomeTab", className="tab", style={'background-color':'#282828', 'color':'white'}),
        dcc.Tab(label="1. Neighbourhoods", value="Figure1Tab", className="tab", style={'background-color':'#282828', 'color':'white'}),
        dcc.Tab(label="2. Amenities", value="Figure2Tab", className="tab", style={'background-color':'#282828', 'color':'white'}),
        dcc.Tab(label="3. Ranking", value="Figure3Tab", className="tab", style={'background-color':'#282828', 'color':'white'}) 
    ]),
    html.Div(id="tab_content", children={})
])

@app.callback(Output('tab_content', 'children'),
            Input('global_header_tabs', 'value'))
def render_content(tab):
    if tab == "HomeTab":
        return generate_homepage_layout()
    elif tab == "Figure1Tab":
        return generate_figure1_layout()
    elif tab == "Figure2Tab":
        return generate_figure2_layout()
    elif tab == "Figure3Tab":
        return generate_figure3_layout()

if __name__ == '__main__':
	app.run_server(debug=False)