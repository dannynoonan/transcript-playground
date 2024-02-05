# import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State

dapp = Dash(__name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            requests_pathname_prefix='/tsp_dash/')

# app.layout = html.Div(...)
# ....
# ....
# app layout
dapp.layout = html.Div([
    html.Div(children='Hello World')
])

# @dapp.callback(
#     Output('fig-bar-suppress-state-bias', 'figure'),
#     Output('fig-scatter-dots-suppress-state-bias', 'figure'),
#     Output('fig-scatter-bubbles-suppress-state-bias', 'figure'),
#     Output('fig-map-suppress-state-bias', 'figure'),
#     Input('suppress-state-bias-year-input', 'value'))

if __name__ == "__main__":
    # dapp.run(debug=True)
    dapp.run_server(debug=True)