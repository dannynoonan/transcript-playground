import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash_new.components import generate_navbar


def generate_content(episode_dropdown_options: list, episode: dict, speaker_dropdown_options: list, emotion_dropdown_options: list) -> html.Div:
    content = html.Div([
        generate_navbar(episode_dropdown_options, episode),
        dbc.Card(className="bg-dark", children=[
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=4, children=[
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
                            "Season ", episode['season'], ", Episode ", episode['sequence_in_season'], ": \"", episode['title'], "\" " , 
                            html.Nobr(episode['air_date'][:10])]),
                        html.H3(className="text-white", children=[
                            episode['scene_count'], " scenes, ", episode['line_count'], " lines, ", episode['word_count'], " words"]),
                        html.P(className="text-white", children=['<<  Previous episode  |  Next episode  >>']),
                        html.Div([
                            html.Img(src=f"/static/wordclouds/TNG/TNG_{episode['episode_key']}.png", width='100%',
                                     style={"padding-left": "10px", "padding-top": "5px"}
                            ),
                        ]),
                    ]),
                    dbc.Col(md=8, children=[
                        html.H3("Similar episodes"),
                        html.Div([
                            dcc.Graph(id="episode-similarity-scatter"),
                        ]),
                        html.Br(),
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
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=12, children=[
                        html.H3("Episode timeline"),
                        dbc.Tabs(className="nav nav-tabs", children=[
                            dbc.Tab(label="By speaker dialog", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row(justify="evenly", children=[
                                    dcc.Graph(id="episode-dialog-timeline-new"),
                                ]),
                                dbc.Row([
                                    dbc.Col(md=2, style={'textAlign': 'center'}, children=[
                                        dcc.Checklist(
                                            id="show-layers",
                                            # className="text-white", 
                                            options=[
                                                {'label': 'Show scenes / locations', 'value': 'scene_locations'}
                                            ],
                                            value=[],
                                            inputStyle={"margin-left": "4px", "margin-right": "4px"}
                                        )
                                    ]),
                                ]),
                            ]),
                            dbc.Tab(label="By scene location", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row(justify="evenly", children=[
                                    dcc.Graph(id="episode-location-timeline-new"),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
            dbc.CardBody([
                html.H3("Character chatter"),
                dbc.Row([
                    dbc.Col(md=4, children=[
                        html.Div([
                            html.Br(),
                            dcc.Graph(id="speaker-episode-frequency-bar-chart-new"),
                        ]),
                        html.Div([
                            "Count by: ",
                            dcc.Dropdown(
                                id="scale-by",
                                options=['scene_count', 'line_count', 'word_count'],
                                value='line_count',
                            )
                        ]),
                    ]),
                    dbc.Col(md=8, children=[
                        html.Div([
                            html.Br(),
                            dcc.Graph(id="speaker-3d-network-graph-new"),
                        ]),
                    ]), 
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
                                options=emotion_dropdown_options,
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
                                value='ALL'
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
                            dcc.Graph(id="episode-speaker-mbti-scatter"),
                        ]),
                        html.Br(),
                        html.Div(id="episode-speaker-mbti-dt"),
                    ]),
                    dbc.Col(md=6, children=[
                        html.Div([
                            dcc.Graph(id="episode-speaker-dnda-scatter"),
                        ]),
                        html.Br(),
                        html.Div(id="episode-speaker-dnda-dt"),
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
                                options=['scaled_score', 'tfidf_score'], 
                                value='tfidf_score',
                            )
                        ]),
                    ]),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(md=6, children=[
                        html.Div([
                            dcc.Graph(id="episode-universal-genres-treemap"),
                        ]),
                        html.Br(),
                        html.Div(id="episode-universal-genres-dt"),
                    ]),
                    dbc.Col(md=6, children=[
                        html.Div([
                            dcc.Graph(id="episode-universal-genres-gpt35-v2-treemap"),
                        ]),
                        html.Br(),
                        html.Div(id="episode-universal-genres-gpt35-v2-dt")
                    ]),     
                ]),
            ]),
            dbc.CardBody([
                html.H3("In-episode search"),
                dbc.Row([
                    dbc.Col(md=4, children=[
                        html.Div([
                            "Find lines containing ",
                            dbc.Input(
                                id="qt", 
                                type="text"
                            ),
                        ]),
                    ]),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(md=12, children=[
                        html.P(children=["Results: ", html.Span(id='out-text')]),
                        html.Div([
                            dcc.Graph(id="episode-search-results-gantt"),
                        ]),
                        # html.Br(),
                        # html.Div(id="episode-search-results-dt"),
                    ]), 
                ]),
            ]),
        ])
    ])

    return content
