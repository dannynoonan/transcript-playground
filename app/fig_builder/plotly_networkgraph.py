import igraph as ig
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import app.fig_builder.fig_helper as fh


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


def build_speaker_chatter_scatter3d(show_key: str, data: dict, scale_by: str, dims: dict = None) -> go.Figure:
    print(f'in build_speaker_chatter_scatter3d show_key={show_key} len(data)={len(data)} scale_by={scale_by} dims={dims}')

    # reference: https://plotly.com/python/v3/3d-network-graph/

    # dimension settings
    height = None
    width = None
    node_min = 8
    node_max = 40
    # line_min = 1
    # line_max = 10
    zoom_scale = 1.5
    hover_truncate = 30
    if dims:
        if 'height' in dims:
            height = dims['height']
        if 'width' in dims:
            width = dims['width']
        if 'node_min' in dims:
            node_min = dims['node_min']
        if 'node_max' in dims:
            node_max = dims['node_max']
        # if 'line_min' in dims:
        #     line_min = dims['line_min']
        # if 'line_max' in dims:
        #     line_max = dims['line_max']
        if 'zoom_scale' in dims:
            zoom_scale = dims['zoom_scale']
        if 'hover_truncate' in dims:
            hover_truncate = dims['hover_truncate']

    # graph edge data and properties
    edges = [(edge['source'], edge['target']) for edge in data['edges']]
    edge_weights = [edge['value'] for edge in data['edges']]
    edge_colors = ['rgb(125,125,125)'] * len(data['edges'])

    # NOTE fell short of implementing edge labels, but here's how: 
    # https://community.plotly.com/t/displaying-edge-labels-of-networkx-graph-in-plotly/39113/2
    # https://stackoverflow.com/questions/74607000/python-networkx-plotly-how-to-display-edges-mouse-over-text/74633879#74633879
    # edges_hovertemplate = "Scenes together: %{customdata[0]}<extra></extra>"
    # edges_customdata = [(ew) for ew in edge_weights]

    # graph node data and properties
    speakers = []
    node_colors = []
    nodes_customdata = []
    for n in data['nodes']:
        speakers.append(n['speaker'])
        node_colors.append(n['color'])
        n['assoc_str'] = ','.join(n['associations'])
        if len(n['assoc_str']) > hover_truncate:
            n['assoc_str'] = f"{n['assoc_str'][:hover_truncate]}..."
        nodes_customdata.append((n['scene_count'], n['line_count'], n['word_count'], n['assoc_str']))

    node_scale_basis = [node[scale_by] for node in data['nodes']]
    node_sizes = fh.scale_values(node_scale_basis, low=node_min, high=node_max)

    node_line = dict(color='rgb(50,50,50)', width=0.5)
    nodes_hovertemplate = "<br>".join([
            "<b>%{text}</b>",
            "Scenes: %{customdata[0]}",
            "Lines: %{customdata[1]}",
            "Words: %{customdata[2]}",
            "Scenes with: %{customdata[3]}",
            "<extra></extra>"
        ])
    
    # graph declaration
    graph = ig.Graph(edges, directed=False)
    graph_layout = graph.layout('kk', dim=3)
    
    # node and edge coordinates
    node_count = len(data['nodes'])
    Xn = [graph_layout[k][0] for k in range(node_count)] # x-coordinates of nodes
    Yn = [graph_layout[k][1] for k in range(node_count)] # y-coordinates
    Zn = [graph_layout[k][2] for k in range(node_count)] # z-coordinates
    Xe = []
    Ye = []
    Ze = []
    # NOTE leaving remnants of flattened array / single trace edge implementation
    # that solution was simpler but did not support variable edge line thickness
    for e in edges:
        Xe.append([graph_layout[e[0]][0], graph_layout[e[1]][0], None]) # x-coordinates of edge ends
        Ye.append([graph_layout[e[0]][1], graph_layout[e[1]][1], None]) # y-coordinates
        Ze.append([graph_layout[e[0]][2], graph_layout[e[1]][2], None]) # z-coordinates
        # Xe.extend([graph_layout[e[0]][0], graph_layout[e[1]][0], None]) 
        # Ye.extend([graph_layout[e[0]][1], graph_layout[e[1]][1], None])
        # Ze.extend([graph_layout[e[0]][2], graph_layout[e[1]][2], None])

    # edge_line = dict(color='rgb(125,125,125)', width=1)

    # iterate over edge data to create distinct scatter3d trace per edge
    # based on: https://stackoverflow.com/questions/58335021/plotly-how-to-set-width-to-specific-line
    edge_traces = []
    for i in range(0, len(Xe)):
        edge_traces.append(go.Scatter3d(x=Xe[i], y=Ye[i], z=Ze[i], line=dict(color=edge_colors[i], width=edge_weights[i]),
                                        mode='lines', hoverinfo='none', opacity=0.5))

    # NOTE original single-trace edge declaration
    # edges_trace = go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', line=edge_line, hoverinfo='none')

    # create one scatter3d trace for all nodes
    nodes_trace = go.Scatter3d(x=Xn, y=Yn, z=Zn, mode='markers', name='speakers', text=speakers, 
                               hovertemplate=nodes_hovertemplate, customdata=nodes_customdata,
                               marker=dict(symbol='circle', size=node_sizes, color=node_colors, colorscale='Viridis', line=node_line))

    axis = dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title='')

    layout = go.Layout(title="Character chatter (scroll to zoom, drag to rotate)", showlegend=False, margin=dict(t=50), hovermode='closest',
        scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis), aspectmode='manual', aspectratio=dict(x=zoom_scale, y=zoom_scale, z=zoom_scale)),
        annotations=[dict(showarrow=False, text='', xref='paper', yref='paper', x=0, y=0.1, xanchor='left', yanchor='bottom', font=dict(size=14))])
    
    # combine trace data and layout into figure
    data = edge_traces + [nodes_trace]
    fig = go.Figure(data=data, layout=layout)

    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(l=20, t=50, r=20, b=20)
    )

    if width:
        fig.update_layout(width=width)
    if height:
        fig.update_layout(height=height)

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
