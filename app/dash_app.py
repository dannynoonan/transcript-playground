# import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State
import pandas as pd

import app.es.es_query_builder as esqb
import app.es.es_response_transformer as esrt
import app.nlp.embeddings_factory as ef
import app.web.fig_builder as fb


dapp = Dash(__name__,
            external_stylesheets=[dbc.themes.SOLAR],
            requests_pathname_prefix='/tsp_dash/')


navbar = dbc.Card(className="text-white bg-primary", style={"z-index":"2000"}, children=[
    dbc.CardBody([
        dbc.Nav(className="nav nav-pills", children=[
            dbc.NavItem(dbc.NavLink("Transcript Playground", style={"color": "#FFFFFF", "font-size": "16pt"}, href="/tsp_dash")),
            # dbc.DropdownMenu(label="Shows", menu_variant="dark", nav=True, children=[
            #     dbc.DropdownMenuItem("TNG", style={"color": "#CCCCCC"}, href='/web/show/TNG', target="_blank"), 
            #     dbc.DropdownMenuItem("GoT", style={"color": "#CCCCCC"}, href='/web/show/GoT', target="_blank"), 
            # ]),
            dbc.NavItem(dbc.NavLink("TNG", style={"color": "#FFFFFF"}, href='/web/show/TNG', external_link=True)),
            dbc.NavItem(dbc.NavLink("Episodes", style={"color": "#CCCCCC"}, href='/web/episode_search/TNG', external_link=True)),
            dbc.NavItem(dbc.NavLink("Character", style={"color": "#CCCCCC"}, href='/web/character_listing/TNG', external_link=True)),
        ])
    ])
])


dapp.layout = html.Div([
    navbar,
    dbc.Card(className="bg-dark", children=[
        dbc.CardBody([
            dbc.Row([
                html.H3(children=["Cluster groupings for ", html.Span(id='show-key-display')]),
                dbc.Col(md=2, children=[
                    html.Div([
                        "Show: ",
                        dcc.Dropdown(
                            id="show-key",
                            options=[
                                {'label': 'TNG', 'value': 'TNG'},
                                {'label': 'GoT', 'value': 'GoT'},
                            ], 
                            value='TNG',
                        )
                    ]),
                ]),
                dbc.Col(md=2, children=[
                    html.Div([
                        "Number of clusters: ",
                        dcc.Dropdown(
                            id="num-clusters",
                            options=[
                                {'label': '2', 'value': '2'},
                                {'label': '3', 'value': '3'},
                                {'label': '4', 'value': '4'},
                                {'label': '5', 'value': '5'},
                                {'label': '6', 'value': '6'},
                                {'label': '7', 'value': '7'},
                                {'label': '8', 'value': '8'},
                                {'label': '9', 'value': '9'},
                                {'label': '10', 'value': '10'},
                            ], 
                            value='6',
                        )
                    ]),
                ]),
            ]),
            html.Br(),
            dbc.Row(justify="evenly", children=[
                dcc.Graph(id="show-cluster-scatter"),
            ]),
        ]),
    ])
])


############ voter-weight-electoral-college-bias-page4 callbacks
@dapp.callback(
    Output('show-cluster-scatter', 'figure'),
    Output('show-key-display', 'children'),
    Input('show-key', 'value'),
    Input('num-clusters', 'value'))    
def render_show_cluster_scatter(show_key: str, num_clusters: int):
    print(f'in render_show_cluster_scatter, show_key={show_key} num_clusters={num_clusters}')
    num_clusters = int(num_clusters)
    vector_field = 'openai_ada002_embeddings'
    # fetch all model/vendor embeddings for show 
    s = esqb.fetch_all_embeddings(show_key, vector_field)
    doc_embeddings = esrt.return_all_embeddings(s, vector_field)
    doc_clusters_df = ef.cluster_docs(doc_embeddings, num_clusters)
    # fetch all episode metadata for show 
    s = esqb.list_episodes_by_season(show_key)
    episodes_by_season = esrt.return_episodes_by_season(s)
    all_episodes = []
    for _, episodes in episodes_by_season.items():
        all_episodes.extend(episodes)
    episodes_df = pd.DataFrame(all_episodes)
    episodes_df['doc_id'] = episodes_df['episode_key'].apply(lambda x: f'{show_key}_{x}')
    # generate scatterplot
    fig_scatter = fb.generate_graph_plotly(doc_clusters_df, episodes_df, show_key, num_clusters)
    return fig_scatter, show_key


if __name__ == "__main__":
    # dapp.run(debug=True)
    dapp.run_server(debug=True)
