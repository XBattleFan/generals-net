
from agent import Agent
import collections
import const
from feature4 import Pred
import gc
import glob
from keras.models import load_model
import helpers
from model import make_model, symmetrize_batch
import numpy as np
import os
import random
import replay
import sys


np.random.seed(const.SEED)

BUFFER_SIZE = 5000
BATCH_SIZE = 32
PROB_SAMPLE = 0.1
MAX_TURNS = 1000
NUM_EPISODES = 10000
ETA = 1e-3
MAX_STALE_TURNS = 50
EPS = 1e-8


def run_episodes(replay_paths, model, output_folder, reward_type, gamma):
    q = collections.deque()
    for ep in range(NUM_EPISODES):
        print "Episode", ep
        path = np.random.choice(replay_paths)
        sim, _ = replay.parse(path)
        players = range(sim.num_players)

        old_player = np.random.choice(players)
        agents = []
        model_paths = glob.glob(os.path.join(output_folder, "model.*.h5"))
        for i in players:
            if i == old_player and model_paths:
                path = np.random.choice(model_paths)
                print "Using model", path, "for player", old_player
                agents.append(Agent(load_model(path)), Pred(), 'egreedy'))
            else:
                agents.append(Agent(model, Pred(), 'egreedy'))

        states_actions = {i: [] for i in players}
        rewards = {i: [] for i in players}
        for turn in range(MAX_TURNS):
            observations = sim.get_observations()
            if any(obs['done'] for obs in observations.values()):
                break

            moves = {}
            for i in players:
                obs = observations[i]
                state, action_index, policy, moves[i] = agents[i].step(obs)
                if action_index is None:
                    continue

                action = np.zeros(const.NUM_ACTIONS)
                action[action_index] = 1.0
                states_actions[i].append((state, action, policy))

            prev_lands = sim.get_lands()
            prev_units = sim.get_units()
            sim.step(moves)

            lands = sim.get_lands()
            units = sim.get_units()
            alives = sim.get_alives()
            done = sum(alives) == 1

            if reward_type == 'landratio':
                for i in players:
                    prev = prev_lands[i] / sum(prev_lands)
                    if done:
                        rewards[i].append(alives[i] - prev)
                    else:
                        land_ratio = lands[i] / sum(lands)
                        rewards[i].append(land_ratio - prev)

            elif reward_type == 'win':
                for i in players:
                    rewards[i].append(0 if done else (1 if alives[i] else -1))

            else:
                raise ValueError("Invalid reward function: {}".format(reward_type))

        print "Turns:", turn+1
        print "Lands:", sim.get_lands()

        for i in players:
            advantages = discount_rewards(rewards[i], gamma)
            for (state, action, policy), advantage in zip(states_actions[i], advantages):
                if np.random.random() < PROB_SAMPLE:
                    q.append((state, action, policy, advantage))

        while len(q) > BUFFER_SIZE:
            q.popleft()
        # print "Buffer size:", len(q)

        batch = random.sample(q, min(len(q), BATCH_SIZE))
        x = np.array([state for state, _, _, _ in batch])
        a = np.array([action for _, action, _, _ in batch])
        p = np.array([policy for _, _, policy, _ in batch])
        r = np.array([advantage for _, _, _, advantage in batch])

        r = r.astype(float)
        r -= r.mean()
        r /= r.std() + EPS
        r = r[:, np.newaxis]

        grad = a - p
        y = p + ETA * r * grad
        x = list(np.swapaxes(x, 0, 1))
        x, y = symmetrize_batch(x, y)
        model.train_on_batch(x, y)

        if ep > 0 and ep % 10 == 0:
            path = os.path.join(output_folder, "model.{:04}.h5".format(ep))
            model.save(path)

        sys.stdout.flush()
        gc.collect()


def discount_rewards(x, gamma):
    returns = []
    c = 0.0
    for i in range(len(x))[::-1]:
        c *= gamma
        c += x[i]
        returns.append(c)
    return np.array(returns[::-1])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", required=True)
    parser.add_argument("--model")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--reward", required=True)
    parser.add_argument("--gamma", required=True, default=0.98, type=float)
    args = parser.parse_args()

    with open(args.replays, 'r') as f:
        replay_paths = [line.strip() for line in f]

    if args.model:
        model = load_model(args.model)
    else:
        model = make_model()

    run_episodes(replay_paths, model, args.output, args.reward, args.gamma)

