import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import app.data_service.animation_publisher as ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    parser.add_argument("--fig_type", "-f", help="Figure type", required=True)
    parser.add_argument("--span_granularity", "-g", help="Span granularity", required=False)
    parser.add_argument("--season", "-n", help="Season", required=False)
    args = parser.parse_args()
    
    show_key = args.show_key
    fig_type = args.fig_type
    span_granularity = args.span_granularity
    season = args.season

    ap.publish_animation(show_key, fig_type, span_granularity=span_granularity, season=season)


if __name__ == '__main__':
    main()
