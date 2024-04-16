import argparse
import os
import pandas as pd

import app.web.fig_builder as fb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    parser.add_argument("--fig_type", "-f", help="Figure type", required=True)
    parser.add_argument("--season", "-e", help="Season", required=False)
    parser.add_argument("--span_granularity", "-g", help="Span granularity", required=False)
    args = parser.parse_args()
    show_key = args.show_key
    fig_type = args.fig_type
    season = args.season
    span_granularity = args.span_granularity

    valid_fig_types_to_df_sources = {'speaker_frequency_bar': 'speaker_episode_aggs'}
    valid_span_granularities = ['word', 'line', 'scene', 'episode']

    if fig_type not in valid_fig_types_to_df_sources:
        raise Exception(f'Failed to publish animation: fig_type={fig_type} must be in {valid_fig_types_to_df_sources.keys()}')

    # fetch or generate aggregate speaker data and build speaker frequency bar chart
    df_source = valid_fig_types_to_df_sources[fig_type]
    file_path = f'./app/data/{df_source}_{show_key}.csv'
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
        print(f'Loading dataframe for fig_type={fig_type} df_source={df_source} using file_path={file_path}')
    else:
        raise Exception(f'Failed to publish animation: unable to fetch dataframe for fig_type={fig_type} df_source={df_source} using file_path={file_path}')

    # generate and store animation html for fig_type
    if fig_type == 'speaker_frequency_bar':
        if not span_granularity:
            raise Exception(f'Failed to publish animation: `span_granularity` is required for fig_type={fig_type}')
        if span_granularity not in valid_span_granularities:
            raise Exception(f'Failed to publish animation: span_granularity={span_granularity} must be in {valid_span_granularities}')
        
        if season:
            fig = fb.build_speaker_frequency_bar(show_key, df, span_granularity, True, int(season), animate=True)
            output_path = f'app/animations/{fig_type}_{show_key}_S{season}_{span_granularity}.html'
            fig.write_html(output_path, auto_play=False)
            print(f'Successfully generated and saved animation html file to output_path={output_path}')

        else:
            fig = fb.build_speaker_frequency_bar(show_key, df, span_granularity, False, None, animate=True)
            output_path = f'app/animations/{fig_type}_{show_key}_{span_granularity}.html'
            fig.write_html(output_path, auto_play=False)
            print(f'Successfully generated and saved animation html file to output_path={output_path}')


if __name__ == '__main__':
    main()
