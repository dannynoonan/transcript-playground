# import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State

import app.es.es_query_builder as esqb
import app.es.es_response_transformer as esrt
import app.nlp.embeddings_factory as ef
import app.web.data_viz as viz

dapp = Dash(__name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            requests_pathname_prefix='/tsp_dash/')

dapp.layout = html.Div([
    html.H1(children=["Cluster groupings for ", html.Span(id='show-key-display')]),
    html.Div([
        "Show: ",
        dcc.RadioItems(
            options=['TNG', 'GoT'],
            value='TNG',
            id='show-key'
        )
    ]),
    html.Div([
        "Number of clusters: ",
        dcc.RadioItems(
            options=['2', '4', '6', '8', '10'],
            value='6',
            id='num-clusters'
        )
    ]),
    html.Br(),
    dcc.Graph(id="show-cluster-scatter"),
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
    # if not num_clusters:
    #     num_clusters = 4
    vector_field = 'openai_ada002_embeddings'
    # fetch all model/vendor embeddings for show 
    s = esqb.fetch_all_embeddings(show_key, vector_field)
    doc_embeddings = esrt.return_all_embeddings(s, vector_field)
    doc_clusters_df = ef.cluster_docs(doc_embeddings, num_clusters)
    fig_scatter = viz.generate_graph_plotly(doc_clusters_df, show_key, num_clusters)
    print(f'type(fig_scatter)={type(fig_scatter)}')
    return fig_scatter, show_key

if __name__ == "__main__":
    # dapp.run(debug=True)
    dapp.run_server(debug=True)