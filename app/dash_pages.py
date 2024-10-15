import dash
from dash import html, Dash, DiskcacheManager
import dash_bootstrap_components as dbc
import diskcache
from uuid import uuid4


# https://dash.plotly.com/background-callback-caching recommends for local dev only, but may work for prod MVP
launch_uid = uuid4()
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(
    cache, cache_by=[lambda: launch_uid], expire=60
)


dash_pages_app = Dash(__name__,
                      use_pages=True,
                      background_callback_manager=background_callback_manager,
                      external_stylesheets=[dbc.themes.SOLAR],
                      requests_pathname_prefix='/dash_pages/')


if __name__ == "__main__":
    dash_pages_app.run(debug=True)


# app layout
dash_pages_app.layout = html.Div([
    dash.page_container
])
