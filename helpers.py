
import const
import numpy as np


POLICY_SHAPE = const.BOARD_SIZE, const.BOARD_SIZE, const.NUM_DIRECTIONS


def get_action_index(start, end, width):
    start_y, start_x = start // width, start % width
    end_y, end_x = end // width, end % width
    dy = end_y - start_y
    dx = end_x - start_x

    directions = {k: i for i, k in
                  enumerate(zip(const.DY, const.DX))}
    direction = directions[(dy, dx)]
    index = start_y, start_x, direction
    return np.ravel_multi_index(index, POLICY_SHAPE)


def get_move(action_index, width):
    start_y, start_x, direction = np.unravel_index(action_index, POLICY_SHAPE)
    end_y = start_y + const.DY[direction]
    end_x = start_x + const.DX[direction]
    start = start_y * width + start_x
    end = end_y * width + end_x
    # print start_y, start_x, "->", end_y, end_x
    return start, end, False, False

