import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash_new.components import generate_navbar


def generate_content(episode_dropdown_options: list, episode: dict, speaker_dropdown_options: list) -> html.Div:
    content = html.Div([
        generate_navbar(episode_dropdown_options, episode),
        dbc.Card(className="bg-dark", children=[
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=5, children=[
                        dbc.Row([ 
                            dbc.Col(md=6, children=[
                                html.Div([
                                    "Show: ",
                                    dcc.Dropdown(
                                        id="show-key",
                                        options=['TNG', 'GoT'],
                                        value='TNG',
                                    )
                                ]),
                            ]),
                            dbc.Col(md=6, children=[
                                html.Div([
                                    "Episode key: ",
                                    dcc.Dropdown(
                                        id="episode-key",
                                        options=episode_dropdown_options,
                                        value=episode['episode_key'],
                                    )
                                ]),
                            ]),
                        ]),
                        html.Br(),
                        html.H3(className="text-white", children=[
                            "Season ", episode['season'], ", Episode ", episode['sequence_in_season'], ": \"", episode['title'], "\" (", episode['air_date'][:10], ")"]),
                        html.H3(className="text-white", children=[
                            episode['scene_count'], " scenes, ", episode['line_count'], " lines, ", episode['word_count'], " words"]),
                        html.P(className="text-white", children=['<<  Previous episode  |  Next episode  >>']),
                        html.Br(),
                        html.P("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam aliquam massa orci, sit amet ultricies neque placerat sit amet. Nullam pellentesque massa vitae lectus placerat fermentum. Morbi dapibus rhoncus purus, nec posuere metus condimentum ac. In congue leo sit amet condimentum faucibus. Integer nec diam fermentum nulla commodo imperdiet viverra et libero. Curabitur dignissim metus non ex cursus, quis egestas lacus vestibulum. Maecenas efficitur varius ex in imperdiet. Sed pharetra tellus quis neque efficitur, quis varius mi semper. Nulla nisi velit, egestas eu rhoncus id, porta vitae nibh. Quisque id dictum eros. Nam sed lorem rutrum, faucibus tortor et, scelerisque dui. Nullam porta tortor quis erat luctus placerat."),
                        html.Br(),
                        html.P("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam aliquam massa orci, sit amet ultricies neque placerat sit amet. Nullam pellentesque massa vitae lectus placerat fermentum. Morbi dapibus rhoncus purus, nec posuere metus condimentum ac. In congue leo sit amet condimentum faucibus."),
                    ]),
                    dbc.Col(md=3, children=[
                        html.Div([
                            html.Br(),
                            dcc.Graph(id="speaker-episode-frequency-bar-chart-new"),
                        ]),
                        html.Div([
                            "Count by: ",
                            dcc.Dropdown(
                                id="span-granularity",
                                options=['scene_count', 'line_count', 'word_count'],
                                value='line_count',
                            )
                        ]),
                    ]),
                    dbc.Col(md=4, children=[
                        html.Div([
                            html.Br(),
                            dcc.Graph(id="speaker-3d-network-graph-new"),
                        ]),
                    ]), 
                ]),
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=5, children=[
                        html.H3("Character dialog timeline"),
                    ]),
                ]),
                dbc.Row(justify="evenly", children=[
                    dcc.Graph(id="episode-dialog-timeline-new"),
                ]),
                html.Br(),
                html.H3("Scene location timeline"),
                html.Br(),
                dbc.Row(justify="evenly", children=[
                    dcc.Graph(id="episode-location-timeline-new"),
                ]),
            ]),
            dbc.CardBody([
                html.H3("Character sentiment timeline"),
                dbc.Row([
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Freeze on ",
                            dcc.Dropdown(
                                id="freeze-on",
                                options=['emotion', 'speaker'],
                                value='emotion',
                            )
                        ]),
                    ]),
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Emotion ",
                            dcc.Dropdown(
                                id="emotion",
                                options=['Joy', 'Love', 'Empathy', 'Curiosity', 'Sadness', 'Anger', 'Fear', 'Disgust', 'Surprise', 'Confusion'],
                                value='Joy',
                            )
                        ]),
                    ]),
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Speaker ",
                            dcc.Dropdown(
                                id="speaker",
                                options=speaker_dropdown_options,
                            )
                        ]),
                    ]),
                ]),
                html.Br(),
                dbc.Row(justify="evenly", children=[
                    dcc.Graph(id="sentiment-line-chart-new"),
                ]),
            ]),
            dbc.CardBody([
                html.H3("Character personalities during episode"),
                dbc.Row(justify="evenly", children=[
                    dbc.Col(md=6, children=[
                        html.Div([
                            html.Br(),
                            dcc.Graph(id="episode-speaker-mbti-scatter"),
                        ]),
                    ]),
                    dbc.Col(md=6, children=[
                        html.Div([
                            html.Br(),
                            dcc.Graph(id="episode-speaker-dnda-scatter"),
                        ]),
                    ]),       
                ]),
            ]),
            dbc.CardBody([
                html.H3("Episode topic distributions"),
                dbc.Row([
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Score type ",
                            dcc.Dropdown(
                                id="topic-score-type",
                                options=['raw_score', 'score', 'tfidf_score'], 
                                value='tfidf_score',
                            )
                        ]),
                    ]),
                ]),
                dbc.Row([
                    dbc.Col(md=6, children=[
                        html.Div([
                            html.Br(),
                            dcc.Graph(id="episode-universal-genres-treemap"),
                        ]),
                    ]),
                    dbc.Col(md=6, children=[
                        html.Div([
                            html.Br(),
                            dcc.Graph(id="episode-universal-genres-gpt35-v2-treemap"),
                        ]),
                    ]),     
                ]),
            ]),
            dbc.CardBody([
                html.H3("Similar episodes"),
                dbc.Row([
                    dbc.Col(md=2, children=[
                        html.Div([
                            "MLT type ",
                            dcc.Dropdown(
                                id="mlt-type",
                                options=['tfidf', 'openai_embeddings'], 
                                value='tfidf',
                            )
                        ]),
                    ]),
                ]),
                dbc.Row([
                    dbc.Col(md=8, children=[
                        html.Div([
                            html.Br(),
                            dcc.Graph(id="episode-similarity-scatter"),
                        ]),
                    ]),
                    dbc.Col(md=4, children=[
                        html.Div([
                            html.Br(),
                            html.Img(src=f"/static/wordclouds/TNG/TNG_{episode['episode_key']}.png", width='100%',
                                     style={"padding-left": "10px", "padding-top": "5px"}
                            ),
                        ]),
                    ]),
                ]),
            ]),
        ])
    ])

    return content