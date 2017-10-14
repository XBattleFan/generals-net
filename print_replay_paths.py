
import os
import simplejson as json


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--num-players", required=True, type=int)
    parser.add_argument("--min-stars", required=True, type=int)
    parser.add_argument("--reverse", action='store_true')
    args = parser.parse_args()

    for filename in os.listdir(args.folder):
        path = os.path.abspath(os.path.join(args.folder, filename))
        with open(path, 'r') as f:
            d = json.load(f)
        stars = d['stars']
        if not stars or len(stars) != args.num_players:
            continue

        ok = min(stars) >= args.min_stars
        if ok != args.reverse:
            print path

