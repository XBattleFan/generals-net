
from collections import deque
import const
import helpers
import numpy as np


# TODO add last_move_start and last_move_end

DIST_CAP = 100
# LAST_MOVE_CAP = 100
BOARD_SHAPE = const.BOARD_SIZE, const.BOARD_SIZE


class Pred(object):
    def __init__(self):
        self.seen_obs = False

    def setup(self, obs):
        self.shape = obs['height'], obs['width']
        self.width = obs['width']

        self.egen = np.zeros(BOARD_SHAPE, dtype=bool)
        self.seentiles = np.zeros(BOARD_SHAPE, dtype=bool)
        self.cities = np.zeros(BOARD_SHAPE, dtype=bool)

        tiles = np.array(obs['tiles']).reshape(self.shape)
        tiles = self.expand(tiles, const.MOUNTAIN)
        self.passable = (tiles != const.MOUNTAIN) & (tiles != const.OBSTACLE)

        p = obs['player']
        g = obs['generals'][p]
        self.mgen = np.zeros(BOARD_SHAPE, dtype=bool)
        self.mgen[self.yx(g)] = False
        self.mgendist = DIST_CAP - self.bfs([self.yx(g)])

        self.egendist = np.zeros(BOARD_SHAPE, dtype=int)
        self.ecities = np.zeros(BOARD_SHAPE, dtype=bool)
        self.ecitiesdist = np.zeros(BOARD_SHAPE, dtype=int)
        self.first_contact = False
        self.econtactlist = []
        self.econtactdist = np.zeros(BOARD_SHAPE, dtype=int)

        corners = [
            (0, 0),
            (0, obs['width'] - 1),
            (obs['height'] - 1, 0),
            (obs['height'] - 1, obs['width'] - 1),
        ]
        maxcornerdist = np.max([self.bfs([x]) for x in corners], axis=0)
        center = maxcornerdist == maxcornerdist.min()
        m = zip(*np.where(center))
        self.centerdist = DIST_CAP - self.bfs(m)
        self.sumcornerdist = len(corners) * DIST_CAP - np.sum([self.bfs([x]) for x in corners], axis=0)

        self.prev_tiles = tiles
        self.prev_armies = np.zeros(BOARD_SHAPE, dtype=int)

        self.city_units = [0] * len(obs['generals'])
        self.prev_units = obs['scores_units']

        # self.last_move_start = np.zeros(BOARD_SHAPE, dtype=int)
        # self.last_move_end = np.zeros(BOARD_SHAPE, dtype=int)

    def step(self, obs, prev_move):
        if not self.seen_obs:
            self.seen_obs = True
            self.setup(obs)

        p = obs['player']
        tiles = np.array(obs['tiles']).reshape(self.shape)
        tiles = self.expand(tiles, const.MOUNTAIN)
        armies = np.array(obs['armies']).reshape(self.shape)
        armies = self.expand(armies, 0)
        e = (tiles >= 0) & (tiles != p)

        for i, x in enumerate(obs['generals']):
            yx = self.yx(x)
            if i != p and x >= 0 and not self.egen[yx]:
                self.egen[yx] = True
                self.egendist = DIST_CAP - self.bfs([yx])

        for x in obs['cities']:
            yx = self.yx(x)
            self.cities[yx] = True
            if e[yx] and not self.ecities[yx]:
                self.ecities[yx] = True
                m = zip(*np.where(self.ecities))
                self.ecitiesdist = DIST_CAP - self.bfs(m)

        visible = (tiles != const.FOG) & (tiles != const.OBSTACLE)
        visible = self.expand(visible, False)
        self.seentiles[visible] = True

        m = zip(*np.where(e))
        edist = DIST_CAP - self.bfs(m)

        mindist = (self.mgendist * e).max()
        m = zip(*np.where(e & (self.mgendist == mindist)))
        nearedist = DIST_CAP - self.bfs(m)

        units = obs['scores_units'][1-p] - obs['scores_lands'][1-p]
        m = zip(*np.where(e & (armies >= units/3)))
        bigedist = DIST_CAP - self.bfs(m)

        if not self.first_contact and e.sum():
            self.first_contact = True
            self.econtactlist = zip(*np.where(e))
            self.econtactdist = DIST_CAP - self.bfs(self.econtactlist)

        maybe_enemy = self.passable & (tiles != p) & (tiles != const.EMPTY)
        d = self.bfs(self.econtactlist, passable=maybe_enemy)
        m = zip(*np.where((d > 5) & (d <= 10)))
        epredictdist = DIST_CAP - self.bfs(m)

        # m = zip(*np.where(tiles == const.FOG))
        # fogdist = DIST_CAP - self.bfs(m)

        # m = zip(*np.where(~self.seentiles))
        # unseendist = DIST_CAP - self.bfs(m)

        m = zip(*np.where(tiles == p))
        mdist = self.bfs(m)
        m = zip(*np.where(mdist == mdist.max()))
        mfardist = DIST_CAP - self.bfs(m)

        m = zip(*np.where(self.seentiles))
        seendist = self.bfs(m)
        m = zip(*np.where(seendist == seendist.max()))
        seenfardist = DIST_CAP - self.bfs(m)

        units = obs['scores_units']
        for i in range(len(obs['generals'])):
            self.city_units[i] += max(0, self.prev_units[i] - units[i])
        self.prev_units = units

        move_start = np.zeros(BOARD_SHAPE)
        move_end = np.zeros(BOARD_SHAPE)
        if prev_move:
            start, end, _, _ = prev_move
            width = self.width
            start_y, start_x = start // width, start % width
            end_y, end_x = end // width, end % width
            move_start[start_y, start_x] = 1
            move_end[end_y, end_x] = 1

            # self.last_move_start[start_y, start_x] = LAST_MOVE_CAP
            # self.last_move_end[end_y, end_x] = LAST_MOVE_CAP

        # self.last_move_start = np.maximum(self.last_move_start - 1, 0)
        # self.last_move_end = np.maximum(self.last_move_end - 1, 0)

        x = []
        for dy, dx in zip(const.DY, const.DX):
            tiles2 = self.shift(tiles, -dy, -dx, const.MOUNTAIN)
            armies2 = self.shift(armies, -dy, -dx, 0)

            mgen2 = self.shift(self.mgen, -dy, -dx, False)
            egen2 = self.shift(self.egen, -dy, -dx, False)
            seentiles2 = self.shift(self.seentiles, -dy, -dx, False)
            cities2 = self.shift(self.cities, -dy, -dx, False)
            ecities2 = self.shift(self.ecities, -dy, -dx, False)
            mgendist2 = self.shift(self.mgendist, -dy, -dx, 0)
            egendist2 = self.shift(self.egendist, -dy, -dx, 0)
            edist2 = self.shift(edist, -dy, -dx, 0)
            ecities2 = self.shift(self.ecities, -dy, -dx, False)
            ecitiesdist2 = self.shift(self.ecitiesdist, -dy, -dx, 0)
            nearedist2 = self.shift(nearedist, -dy, -dx, 0)
            bigedist2 = self.shift(bigedist, -dy, -dx, 0)
            econtactdist2 = self.shift(self.econtactdist, -dy, -dx, 0)
            epredictdist2 = self.shift(epredictdist, -dy, -dx, 0)
            mfardist2 = self.shift(mfardist, -dy, -dx, 0)
            seenfardist2 = self.shift(seenfardist, -dy, -dx, 0)
            sumcornerdist2 = self.shift(self.sumcornerdist, -dy, -dx, 0)
            centerdist2 = self.shift(self.centerdist, -dy, -dx, 0)
            prev_tiles2 = self.shift(self.prev_tiles, -dy, -dx, const.MOUNTAIN)
            prev_armies2 = self.shift(self.mgendist, -dy, -dx, 0)
            move_start2 = self.shift(move_start, -dy, -dx, 0)
            move_end2 = self.shift(move_end, -dy, -dx, 0)

            f = self.step_direction(obs,
                                    tiles, tiles2,
                                    armies, armies2,
                                    self.mgen, mgen2,
                                    self.egen, egen2,
                                    self.seentiles, seentiles2,
                                    self.cities, cities2,
                                    self.mgendist, mgendist2,
                                    self.egendist, egendist2,
                                    edist, edist2,
                                    self.ecities, ecities2,
                                    self.ecitiesdist, ecitiesdist2,
                                    nearedist, nearedist2,
                                    bigedist, bigedist2,
                                    self.econtactdist, econtactdist2,
                                    epredictdist, epredictdist2,
                                    mfardist, mfardist2,
                                    seenfardist, seenfardist2,
                                    self.sumcornerdist, sumcornerdist2,
                                    self.centerdist, centerdist2,
                                    self.prev_tiles, prev_tiles2,
                                    self.prev_armies, prev_armies2,
                                    move_start, move_start2,
                                    move_end, move_end2)
            x.append(list(f))

        self.prev_tiles = tiles
        self.prev_armies = armies

        x = np.array(x)

        # dimensions are: direction, feature, y, x
        # want: direction, y, x, feature
        assert x.ndim == 4
        assert x.shape[0] == const.NUM_DIRECTIONS
        assert x.shape[2] == const.BOARD_SIZE
        assert x.shape[3] == const.BOARD_SIZE
        x = np.swapaxes(x, 1, 2)
        x = np.swapaxes(x, 2, 3)
        return x

    def step_direction(self, obs,
                       tiles1, tiles2,
                       armies1, armies2,
                       mgen1, mgen2,
                       egen1, egen2,
                       seentiles1, seentiles2,
                       cities1, cities2,
                       mgendist1, mgendist2,
                       egendist1, egendist2,
                       edist1, edist2,
                       ecities1, ecities2,
                       ecitiesdist1, ecitiesdist2,
                       nearedist1, nearedist2,
                       bigedist1, bigedist2,
                       econtactdist1, econtactdist2,
                       epredictdist1, epredictdist2,
                       mfardist1, mfardist2,
                       seenfardist1, seenfardist2,
                       sumcornerdist1, sumcornerdist2,
                       centerdist1, centerdist2,
                       prev_tiles1, prev_tiles2,
                       prev_armies1, prev_armies2,
                       move_start1, move_start2,
                       move_end1, move_end2):
        ones = np.ones(BOARD_SHAPE)

        turn = obs['turn']
        turn_round, turn_mod = turn // 50, turn % 50
        yield ones * (turn_round == 0)
        yield ones * turn_round
        yield ones * turn_mod
        yield ones * (turn_round == 0)
        yield ones * (turn_round == 1)
        yield ones * (turn_round == 2)
        yield ones * (turn_round == 3)
        yield ones * (turn_round > 3)
        yield ones * (turn_mod < 30)
        yield ones * (30 <= turn_mod) * (turn_mod < 35)
        yield ones * (35 <= turn_mod) * (turn_mod < 40)
        yield ones * (40 <= turn_mod) * (turn_mod < 45)
        yield ones * (45 <= turn_mod)

        p = obs['player']
        m1 = tiles1 == p
        m2 = tiles2 == p
        e1 = (tiles1 >= 0) & (tiles1 != p)
        e2 = (tiles2 >= 0) & (tiles2 != p)
        yield m1
        yield m2
        yield e1
        yield e2

        marmies1 = m1 * armies1
        marmies2 = m2 * armies2
        earmies1 = e1 * armies2
        earmies2 = e2 * armies2
        narmies2 = (~m2) * (~e2) * armies2
        for x in [marmies1, marmies2, earmies1, earmies2]:
            yield x
            yield x > 1
            yield x == 1
            yield x == 2
            yield x == 3
            yield x == 4
            yield (5 <= x) & (x < 10)
            yield 10 <= x
            yield x == x.max()
            yield x == x.max() & (x >= 10)
            yield x >= (x.max() / 2) & (x >= 10)

        yield narmies2

        passable1 = (tiles1 != const.MOUNTAIN) & (tiles1 != const.OBSTACLE)
        passable2 = (tiles2 != const.MOUNTAIN) & (tiles2 != const.OBSTACLE)
        visible1 = (tiles1 != const.FOG) & (tiles1 != const.OBSTACLE)
        visible2 = (tiles2 != const.FOG) & (tiles2 != const.OBSTACLE)
        yield passable1
        yield passable2
        yield visible1
        yield visible2
        yield seentiles1
        yield seentiles2

        valid_move = (marmies1 > 1) & passable2
        can_take = marmies1 > armies2 + 1
        yield valid_move
        yield e2 & can_take
        yield e2 & (~can_take)

        mcities1 = m1 & cities1
        mcities2 = m2 & cities2
        ncities1 = (~m1) & (~ecities1) & cities1
        ncities2 = (~m2) & (~ecities2) & cities2
        yield mcities1
        yield mcities2
        yield ecities1
        yield ecities2 & can_take
        yield ecities2 & (~can_take)
        yield ncities1
        yield ncities2 & can_take
        yield ncities2 & (~can_take)

        yield mgen1
        yield mgen2
        yield egen1
        yield egen2 * can_take
        yield egen2 * (~can_take)

        for d1, d2 in [
            (mgendist1, mgendist2),
            (egendist1, egendist2),
            (edist1, edist2),
            (ecities1, ecities2),
            (ecitiesdist1, ecitiesdist2),
            (nearedist1, nearedist2),
            (bigedist1, bigedist2),
            (econtactdist1, econtactdist2),
            (epredictdist1, epredictdist2),
            (mfardist1, mfardist2),
            (seenfardist1, seenfardist2),
            (sumcornerdist1, sumcornerdist2),
            (centerdist1, centerdist2),
        ]:
            yield d1
            yield d2
            yield (d1 > d2) & (d2 > 0)
            yield (d1 < d2) & (d1 > 0)

        lands = obs['scores_lands']
        units = obs['scores_units']
        yield ones * (lands[p] - lands[1-p])
        yield ones * (units[p] - lands[p] - units[1-p] + lands[1-p])
        yield ones * (self.city_units[p] - self.city_units[1-p])

        prev_m1 = prev_tiles1 == p
        prev_m2 = prev_tiles2 == p
        prev_e1 = (prev_tiles1 >= 0) & (prev_tiles1 != p)
        prev_e2 = (prev_tiles2 >= 0) & (prev_tiles2 != p)
        prev_marmies1 = prev_m1 * prev_armies1
        prev_marmies2 = prev_m2 * prev_armies2
        prev_earmies1 = prev_e1 * prev_armies1
        prev_earmies2 = prev_e2 * prev_armies2
        yield marmies1 - prev_marmies1
        yield marmies2 - prev_marmies2
        yield earmies1 - prev_earmies1
        yield earmies2 - prev_earmies2

        yield move_start1
        yield move_start2
        yield move_end1
        yield move_end2

    def yx(self, i):
        return i // self.width, i % self.width

    def bfs(self, initial, passable=None):
        if passable is None:
            passable = self.passable

        dist = np.full(BOARD_SHAPE, DIST_CAP)
        for x in initial:
            dist[x] = 0

        q = deque()
        q.extendleft(initial)
        seen = set(initial)

        while q:
            yx = y, x = q.pop()
            if yx in seen:
                continue
            seen.add(yx)

            for dy, dx in zip(const.DY, const.DX):
                yx2 = y2, x2 = y+dy, x+dx
                if (passable[yx2] and 0 <= y2 < const.BOARD_SIZE
                        and 0 <= x2 < const.BOARD_SIZE):
                    dist[yx2] = dist[yx] + 1
                    q.appendleft(yx2)
        return dist

    def shift(self, x, dy, dx, value):
        height, width = x.shape
        up = np.full((max(dy, 0), width), value)
        x = x[max(-dy, 0):, :]
        x = np.vstack([up, x])
        x = x[:height, :]

        height, width = x.shape
        left = np.full((height, max(dx, 0)), value)
        x = x[:, max(-dx, 0):]
        x = np.hstack([left, x])
        x = x[:, :width]
        return self.expand(x, value)

    def expand(self, x, value):
        n = const.BOARD_SIZE
        height, width = x.shape
        right = np.full((height, n-width), value)
        down = np.full((n-height, n), value)
        x = np.hstack([x, right])
        x = np.vstack([x, down])
        return x


class Dep(object):
    def step(self, obs, action):
        start, end, _, _ = action

        y = np.zeros(const.NUM_ACTIONS)
        if start < 0 or end < 0:
            return y

        i = helpers.get_action_index(start, end, obs['width'])
        y[i] = 1.0
        return y

