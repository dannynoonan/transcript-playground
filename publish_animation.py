import argparse

import app.data_service.animation_publisher as ap


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

    ap.publish_animation(show_key, fig_type, span_granularity=span_granularity, season=season)


if __name__ == '__main__':
    main()
