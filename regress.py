
import const
from feature4 import Pred, Dep
import h5py
import numpy as np
import os
import replay
import simplejson as json
import tqdm
import yaml


MAX_FEATURES = 200
MAX_TURNS = 1000


def regress(replay_paths, output_path):
    maxshape = (None, const.NUM_DIRECTIONS, const.BOARD_SIZE,
                const.BOARD_SIZE, MAX_FEATURES)
    with h5py.File(output_path, 'a') as f:
        f.create_dataset('pred', compression='lzf', shape=(0, 0, 0, 0, 0),
                         maxshape=maxshape)
        f.create_dataset('dep', compression='lzf', shape=(0, 0),
                         maxshape=(None, const.NUM_ACTIONS))

    for path in tqdm.tqdm(replay_paths):
        sim, moves = replay.parse(path)
        preds = [Pred() for _ in range(sim.num_players)]
        deps = [Dep() for _ in range(sim.num_players)]

        x = []
        y = []
        for turn, actions in enumerate(moves[:MAX_TURNS]):
            observations = sim.get_observations()
            if any(obs['done'] for obs in observations.values()):
                break

            for i in range(sim.num_players):
                obs = observations[i]
                prev_move = moves[turn-1][i] if turn else None
                p = preds[i].step(obs, prev_move)
                d = deps[i].step(obs, actions[i])
                if d.sum():
                    x.append(p)
                    y.append(d)

            sim.step(actions)

        x, y = np.array(x), np.array(y)

        if not x.size or not y.size:
            continue

        with h5py.File(output_path, 'a') as f:
            dset = f['pred']
            dset.resize((dset.shape[0] + x.shape[0],) + x.shape[1:])
            dset[-x.shape[0]:] = x

            dset = f['dep']
            dset.resize((dset.shape[0] + y.shape[0],) + y.shape[1:])
            dset[-y.shape[0]:] = y


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    with open(args.replays, 'r') as f:
        replay_paths = [line.strip() for line in f]
    regress(replay_paths, args.output)

