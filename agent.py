
import const
import helpers
import numpy as np


START_OFFSET = 22
RANDOM_MOVE = 0.1


class Agent(object):
    def __init__(self, model, pred, policy):
        self._model = model
        self._pred = pred
        self._policy = policy
        self._prev_move = None

    def step(self, obs):
        state = self._pred.step(obs, self._prev_move)

        # before: batch, direction, y, x, feature
        # after: batch, y, x, direction, feature
        x = np.moveaxis(x, 1, 3)
        x = state[np.newaxis]
        policy = self._model.predict(x).ravel()

        valid = get_actions(obs)
        (valid_actions,) = np.where(valid)
        if obs['turn'] < START_OFFSET or not valid_actions.size:
            return x, None, policy, (-1, -1, False, False)

        if self._policy == 'greedy':
            action_index = (policy * valid).argmax()
        elif self._policy == 'egreedy':
            if np.random.random() < RANDOM_MOVE:
                action_index = np.random.choice(valid_actions)
            else:
                action_index = (policy * valid).argmax()
        elif self._policy == 'sample':
            actions = np.arange(const.NUM_ACTIONS)
            action_index = np.random.choice(actions, p=policy)
        elif self._policy == 'esample':
            if np.random.random() < RANDOM_MOVE:
                action_index = np.random.choice(valid_actions)
            else:
                actions = np.arange(const.NUM_ACTIONS)
                action_index = np.random.choice(actions, p=policy)
        else:
            raise ValueError("Invalid policy")

        move = self._prev_move = helpers.get_move(action_index, obs['width'])
        return state, action_index, policy, move


def get_actions(obs):
    width = obs['width']
    y = np.zeros(const.NUM_ACTIONS)
    n = len(obs['tiles'])
    for start in range(n):
        for end in range(n):
            start_y, start_x = start // width, start % width
            end_y, end_x = end // width, end % width
            if abs(end_y - start_y) + abs(end_x - start_x) != 1:
                continue
            if obs['tiles'][start] != obs['player'] or obs['armies'][start] < 2:
                continue
            if obs['tiles'][end] in [const.MOUNTAIN, const.OBSTACLE]:
                continue

            action_index = helpers.get_action_index(start, end, width)
            y[action_index] = 1
    return y

