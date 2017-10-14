
import const
from sim import Sim
import simplejson as json


# TODO change ReplayAgent into list of move dicts


def parse(path):
    with open(path, 'r') as f:
        replay = json.load(f)

    size = replay['mapHeight'] * replay['mapWidth']

    tiles = [const.EMPTY] * size
    for x in replay['mountains']:
        tiles[x] = const.MOUNTAIN

    armies = [0] * size
    for x, n in zip(replay['cities'], replay['cityArmies']):
        armies[x] = n

    for i, x in enumerate(replay['generals']):
        armies[x] = 1
        tiles[x] = i

    sim = Sim(replay['mapWidth'],
              replay['mapHeight'],
              replay['generals'],
              replay['cities'],
              tiles,
              armies)
    num_players = len(replay['generals'])

    turns = max(move['turn'] for move in replay['moves']) + 1 if replay['moves'] else 0
    moves = [{i: (-1, -1, False, False)
             for i in range(num_players)}
             for _ in range(turns)]

    for move in replay['moves']:
        turn = move['turn']
        player = move['index']
        moves[turn][player] = move['start'], move['end'], move['is50'], False

    for afk in replay['afks']:
        player = afk['index']
        turn = afk['turn'] + 50
        if turn < len(moves):
            moves[turn][player][-1] = True

    return sim, moves


# class ReplayAgent(object):
#     def __init__(self, replay, player):
#         self.moves = {move['turn']: move for move in replay['moves']
#                       if move['index'] == player}
# 
#         self.afk_turn = None
#         for afk in replay['afks']:
#             if afk['index'] == player:
#                 self.afk_turn = afk['turn'] + 50
# 
#     def step(self, obs):
#         turn = obs['turn']
#         is_afk = turn == self.afk_turn
#         if turn in self.moves:
#             move = self.moves[turn]
#             return move['start'], move['end'], move['is50'], is_afk
#         else:
#             return -1, -1, False, is_afk

