from bertopic import BERTopic
import plotly.graph_objects as go


def build_bertopic_visualize_barchart(bertopic_model: BERTopic) -> go.Figure:
    '''
    Generate topic keyword barcharts using saved model file
    '''
    # fig = bertopic_model.visualize_barchart(top_n_topics=16, width=200, height=250)
    fig = bertopic_model.visualize_barchart(top_n_topics=16, width=400, height=300)

    # TODO saving
    # https://maartengr.github.io/BERTopic/api/plotting/barchart.html#bertopic.plotting._barchart.visualize_barchart
    # fig = topic_model.visualize_barchart()
    # fig.write_html("path/to/file.html")

    return fig


def build_bertopic_visualize_topics(bertopic_model: BERTopic) -> go.Figure:
    '''
    Generate topic graphs using saved model file
    '''
    fig = bertopic_model.visualize_topics(width=800, height=800)

    return fig


def build_bertopic_visualize_hierarchy(bertopic_model: BERTopic) -> go.Figure:
    '''
    Generate topic hierarchy using saved model file
    '''
    fig = bertopic_model.visualize_hierarchy(width=1600, height=1200)

    return fig
