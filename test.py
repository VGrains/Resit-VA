import dash_leaflet as dl
from dash import Dash, html

# Cool, dark tiles by Stadia Maps.
url = 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png'
attribution = '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> '
# Create app.
app = Dash()
app.layout = html.Div([
    dl.Map(center=[39, -98], zoom=4, children=[dl.TileLayer(url=url, maxZoom=20, attribution=attribution)])
], style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block", "position": "relative"})

if __name__ == '__main__':
    app.run_server(port=8051, debug=True)