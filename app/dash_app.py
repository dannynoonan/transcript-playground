# import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd

import app.es.es_query_builder as esqb
import app.es.es_response_transformer as esrt
import app.nlp.embeddings_factory as ef
import app.web.fig_builder as fb
import app.web.fig_metadata as fm


# class DataObj(object):
#     def __init__(self):
#         self.display_df = pd.read_csv('clusters_simple_TNG_6.csv')

# data_obj = DataObj()

# def update_data_obj(df: pd.DataFrame) -> None:
#     data_obj.display_df = df

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
                            value='2',
                        )
                    ]),
                ]),
            ]),
            html.Br(),
            dbc.Row(justify="evenly", children=[
                dcc.Graph(id="show-cluster-scatter"),
            ]),
            html.Br(),
            html.Div(id="episodes-df-table"),
            # dash_table.DataTable(data_obj.display_df.to_dict('records'), [{"name": i, "id": i} for i in data_obj.display_df.columns])
        ]),
    ])
])


############ voter-weight-electoral-college-bias-page4 callbacks
@dapp.callback(
    Output('show-cluster-scatter', 'figure'),
    Output('show-key-display', 'children'),
    Output('episodes-df-table', 'children'),
    Input('show-key', 'value'),
    Input('num-clusters', 'value'))    
def render_show_cluster_scatter(show_key: str, num_clusters: int):
    print(f'in render_show_cluster_scatter, show_key={show_key} num_clusters={num_clusters}')
    num_clusters = int(num_clusters)
    vector_field = 'openai_ada002_embeddings'

    # fetch embeddings for all show episodes 
    s = esqb.fetch_all_embeddings(show_key, vector_field)
    doc_embeddings = esrt.return_all_embeddings(s, vector_field)

    # generate and color-stamp clusters for all show episodes 
    doc_embeddings_clusters_df = ef.cluster_docs(doc_embeddings, num_clusters)
    doc_embeddings_clusters_df['cluster_color'] = doc_embeddings_clusters_df['Cluster'].apply(lambda x: fm.colors[x])

    # fetch basic title/season data for all show episodes 
    s = esqb.list_episodes_by_season(show_key)
    episodes_by_season = esrt.return_episodes_by_season(s)
    all_episodes = []
    for _, episodes in episodes_by_season.items():
        all_episodes.extend(episodes)
    episodes_df = pd.DataFrame(all_episodes)

    # merge basic episode data into cluster data
    episodes_df['doc_id'] = episodes_df['episode_key'].apply(lambda x: f'{show_key}_{x}')
    episode_embeddings_clusters_df = pd.merge(doc_embeddings_clusters_df, episodes_df, on='doc_id', how='outer')

    # generate dash_table div as part of callback output
    episode_clusters_df = episode_embeddings_clusters_df[fm.episode_keep_cols + fm.cluster_cols].copy()
    table_div = merge_and_simplify_df(episode_clusters_df)

    # generate scatterplot
    fig_scatter = fb.generate_graph_plotly(episode_embeddings_clusters_df, show_key, num_clusters)

    return fig_scatter, show_key, table_div


# TODO where should this live? Can/will it be more generic, or limited to this use case?
def merge_and_simplify_df(episode_clusters_df: pd.DataFrame) -> html.Div:
    # reformat columns, sort table
    episode_clusters_df['air_date'] = episode_clusters_df['air_date'].apply(lambda x: x[:10])
    episode_clusters_df['focal_speakers'] = episode_clusters_df['focal_speakers'].apply(lambda x: ", ".join(x))
    episode_clusters_df['focal_locations'] = episode_clusters_df['focal_locations'].apply(lambda x: ", ".join(x))
    episode_clusters_df.sort_values(['Cluster', 'season', 'sequence_in_season'], inplace=True)
    # generate table div that can function as an identifiable dash object
    table_div = html.Div([
        dash_table.DataTable(
            data=episode_clusters_df.to_dict("records"),
            columns=[{"id": x, "name": x} for x in episode_clusters_df.columns],
            style_cell={'textAlign': 'left'},
            style_data={
                'color': 'black',
                'backgroundColor': 'white'
            },
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': "{Cluster} = 0",
                        'column_id': 'cluster_color'
                    },
                    'backgroundColor': fm.colors[0],
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': "{Cluster} = 1",
                        'column_id': 'cluster_color'
                    },
                    'backgroundColor': fm.colors[1],
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': "{Cluster} = 2",
                        'column_id': 'cluster_color'
                    },
                    'backgroundColor': fm.colors[2],
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': "{Cluster} = 3",
                        'column_id': 'cluster_color'
                    },
                    'backgroundColor': fm.colors[3],
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': "{Cluster} = 4",
                        'column_id': 'cluster_color'
                    },
                    'backgroundColor': fm.colors[4],
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': "{Cluster} = 5",
                        'column_id': 'cluster_color'
                    },
                    'backgroundColor': fm.colors[5],
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': "{Cluster} = 6",
                        'column_id': 'cluster_color'
                    },
                    'backgroundColor': fm.colors[6],
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': "{Cluster} = 7",
                        'column_id': 'cluster_color'
                    },
                    'backgroundColor': fm.colors[7],
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': "{Cluster} = 8",
                        'column_id': 'cluster_color'
                    },
                    'backgroundColor': fm.colors[8],
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': "{Cluster} = 9",
                        'column_id': 'cluster_color'
                    },
                    'backgroundColor': fm.colors[9],
                    'color': 'white'
                },
            ],
            # style_header={
            #     'backgroundColor': 'rgb(210, 210, 210)',
            #     'color': 'black',
            #     'fontWeight': 'bold'
            # }
        )
    ])
    return table_div


if __name__ == "__main__":
    dapp.run_server(debug=True)
