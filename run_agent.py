
from agent import Agent
from client import Client
import const
from feature4 import Pred
import numpy as np
import time


def run_agent(model, region, userid, mode, gameid, policy):
    client = Client(region, userid, mode, gameid=gameid)
    agent = Agent(model, Pred(), policy)
    for obs in client.get_updates():
        t = time.time()
        print "Turn", obs['turn'] / 2
        for username, units, lands in zip(obs['usernames'], obs['scores_units'],
                                          obs['scores_lands']):
            print username, '\t', units, '\t', lands

        _, _, _, move = agent.step(obs)
        start, end, is_half, _ = move
        client.move(start, end, is_half)
        print "Latency:", int((time.time() - t) / 1000)
        print obs
        print


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--userid", required=True)
    parser.add_argument("--mode", default='private')
    parser.add_argument("--gameid")
    parser.add_argument("--policy", default='greedy')
    args = parser.parse_args()

    from keras.models import load_model
    model = load_model(args.model)
    run_agent(model, args.region, args.userid, args.mode, args.gameid, args.policy)

