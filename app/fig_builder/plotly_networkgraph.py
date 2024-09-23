import igraph as ig
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_bertopic_model_3d_scatter(show_key: str, bertopic_model_id: str, bertopic_docs_df: pd.DataFrame) -> go.Figure:
    print(f'in build_bertopic_model_3d_scatter show_key={show_key} bertopic_model_id={bertopic_model_id}')

    custom_data = ['cluster_title_short', 'cluster', 'season', 'episode', 'title', 'speaker_group', 'topics_focused_tfidf_list']

    bertopic_docs_df['cluster_title_short_legend'] = bertopic_docs_df['cluster'].astype(str) + ' ' + bertopic_docs_df['cluster_title_short']
    # bertopic_docs_df['cluster_title_short_legend'] = bertopic_docs_df[['cluster', 'cluster_title_short']].apply(lambda x: ' '.join(str(x)), axis=1)

    fig = px.scatter_3d(bertopic_docs_df, x='x_coord', y='y_coord', z='z_coord', color='cluster_title_short_legend', opacity=0.7, custom_data=custom_data,
                        # labels={'Topic', 'Topic'}, color_discrete_map=color_discrete_map, category_orders=category_orders,
                        height=1000, width=1600)

    fig.update_traces(marker=dict(line=dict(width=0.1, color='DarkSlateGrey')), selector=dict(mode='markers'))

    fig.update_traces(
        hovertemplate = "".join([
            "<b>%{customdata[0]} (Topic %{customdata[1]}</b><br><br>",
            "<b>S%{customdata[2]}:E%{customdata[3]}: %{customdata[4]}</b><br>",            
            "Speaker group: %{customdata[5]}<br>",
            "Focal topics: %{customdata[6]}",
            "<extra></extra>"
        ]),
        # mode='markers',
        # marker={'sizemode':'area',
        #         'sizeref':10},
    )

    return fig


def build_3d_network_graph(show_key: str, data: dict) -> go.Figure:
    print(f'in build_3d_network_graph show_key={show_key}')

    # reference: https://plotly.com/python/v3/3d-network-graph/
    
    # data = esr.episode_relations_graph(ShowKey(show_key), model_vendor, model_version, max_edges=max_edges)

    node_count = len(data['nodes'])

    edge_count = len(data['links'])
    edges = [(data['links'][k]['source'], data['links'][k]['target']) for k in range(edge_count)]

    graph = ig.Graph(edges, directed=False)

    labels = [node['name'] for node in data['nodes']]
    group = [node['group'] for node in data['nodes']]

    graph_layout = graph.layout('kk', dim=3)

    Xn = [graph_layout[k][0] for k in range(node_count)] # x-coordinates of nodes
    Yn = [graph_layout[k][1] for k in range(node_count)] # y-coordinates
    Zn = [graph_layout[k][2] for k in range(node_count)] # z-coordinates
    Xe = []
    Ye = []
    Ze = []
    for e in edges:
        Xe += [graph_layout[e[0]][0], graph_layout[e[1]][0], None] # x-coordinates of edge ends
        Ye += [graph_layout[e[0]][1], graph_layout[e[1]][1], None]
        Ze += [graph_layout[e[0]][2], graph_layout[e[1]][2], None]

    edge_line = dict(color='rgb(125,125,125)', width=1)
    node_line = dict(color='rgb(50,50,50)', width=0.5)

    edges_trace = go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', line=edge_line, hoverinfo='none')

    nodes_trace = go.Scatter3d(x=Xn, y=Yn, z=Zn, mode='markers', name='actors', text=labels, hoverinfo='text',
        marker=dict(symbol='circle', size=6, color=group, colorscale='Viridis', line=node_line))

    axis = dict(showbackground=False,  showline=False, zeroline=False, showgrid=False, showticklabels=False, title='')

    layout = go.Layout(title="Character chatter", showlegend=False, margin=dict(t=100), hovermode='closest',
        scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
        annotations=[dict(showarrow=False, text="TODO", xref='paper', yref='paper', x=0, y=0.1, xanchor='left', yanchor='bottom', font=dict(size=14))],
        # width=800, height=650,
    )
    
    data = [edges_trace, nodes_trace]
    fig = go.Figure(data=data, layout=layout)

    return fig  


# NOTE this is a template I must have re-purposed at some point
def build_network_graph() -> go.Figure:
    print(f'in build_network_graph')

    # reference https://plotly.com/python/network-graphs/

    G = nx.random_geometric_graph(200, 0.125)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title='<br>Network graph made with Python',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="annotation text",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig
