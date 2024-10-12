import dash
from dash import html, Dash
import dash_bootstrap_components as dbc


dash_pages_app = Dash(__name__,
                      use_pages=True,
                      external_stylesheets=[dbc.themes.SOLAR],
                      requests_pathname_prefix='/dash_pages/')


if __name__ == "__main__":
    dash_pages_app.run(debug=True)


# app layout
dash_pages_app.layout = html.Div([
    dash.page_container
])
