
import const


class Sim(object):
    def __init__(self, width, height,
                 generals, cities, tiles, armies):
        self.width = width
        self.height = height
        self.size = width * height

        self.generals = generals
        self.cities = cities
        self.tiles = tiles
        self.armies = armies

        self.num_players = len(generals)
        self.afks = set()
        self.turn = 0

    def step(self, actions):
        players = range(self.num_players)
        if self.turn % 2:
            players = players[::-1]

        for player in players:
            start, end, is_half, is_afk = actions[player]
            if start >= 0 and end >= 0:
                self._move(player, start, end, is_half)
            if is_afk:
                self.afks.add(player)
        self._end_turn()

    def get_observations(self):
        return {player: self._make_obs(player)
                for player in range(self.num_players)}

    def _move(self, player, start, end, is_half):
        if not (0 <= start < len(self.tiles) and 0 <= end < len(self.tiles)):
            return
        if not (self.tiles[start] == player >= 0):
            return
        if not (self.tiles[end] >= 0 or self.tiles[end] == const.EMPTY):
            return
        # assert self.tiles[start] == player >= 0
        # assert self.tiles[end] >= 0 or self.tiles[end] == const.EMPTY

        if is_half:
            n = self.armies[start] // 2
        else:
            n = self.armies[start] - 1

        self.armies[start] -= n
        m = self.armies[end]

        # merge armies
        if self.tiles[end] == player:
            self.armies[end] += n

        # attack and capture
        elif n > m:
            self.armies[end] = n - m
            tile = self.tiles[end]
            self.tiles[end] = player

            # capture general
            if tile >= 0 and tile != player and self.generals[tile] == end:
                for i in range(self.size):
                    if self.tiles[i] == tile:
                        self.tiles[i] = player
                        self.armies[i] = (self.armies[i] + 1) // 2

        # attack and do not capture
        else:
            self.armies[end] -= n

        assert self.armies[start] >= 0
        assert self.armies[end] >= 0

    def _end_turn(self):
        self.turn += 1
        if self.turn % 2 == 0:
            for i in self.generals:
                if self.tiles[i] not in self.afks:
                    self.armies[i] += 1

            for i in self.cities:
                tile = self.tiles[i]
                if tile >= 0 and tile not in self.afks:
                    self.armies[i] += 1

        if self.turn % 50 == 0:
            for i in range(self.size):
                if self.tiles[i] >= 0:
                    self.armies[i] += 1

    def _make_obs(self, player):
        vis = set()
        for i, p in enumerate(self.tiles):
            if p != player:
                continue
            x, y = i % self.width, i // self.width
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    y2 = y+dy
                    x2 = x+dx
                    if 0 <= x2 < self.width and 0 <= y2 < self.height:
                        vis.add(y2 * self.width + x2)

        generals = [i if i in vis else -1 for i in self.generals]
        cities = [i for i in self.cities if i in vis]
        tiles = self.tiles[:]
        armies = self.armies[:]
        for i in range(self.size):
            if i not in vis:
                if self.tiles[i] == const.MOUNTAIN or i in self.cities:
                    tiles[i] = const.OBSTACLE
                else:
                    tiles[i] = const.FOG
                armies[i] = 0

        scores_lands = []
        scores_units = []
        scores_alives = []
        for p in range(self.num_players):
            scores_lands.append(sum(1 for i in range(self.size)
                                    if self.tiles[i] == p))
            scores_units.append(sum(self.armies[i] for i in range(self.size)
                                     if self.tiles[i] == p))
            scores_alives.append(self.tiles[self.generals[p]] == p)

        players_left = {self.tiles[i] for i in self.generals} - self.afks
        assert players_left

        return {
            'done': len(players_left) == 1,
            'width': self.width,
            'height': self.height,
            'generals': generals,
            'cities': cities,
            'tiles': tiles,
            'armies': armies,
            'turn': self.turn,
            'scores_lands': scores_lands,
            'scores_units': scores_units,
            'scores_alives': scores_alives,
            'player': player,
        }

    def get_lands(self):
        scores_lands = []
        for p in range(self.num_players):
            scores_lands.append(sum(1 for i in range(self.size)
                                    if self.tiles[i] == p))
        return scores_lands


    def get_units(self):
        scores_units = []
        for p in range(self.num_players):
            scores_units.append(sum(self.armies[i] for i in range(self.size)
                                     if self.tiles[i] == p))
        return scores_units

    def get_alives(self):
        scores_alives = []
        for p in range(self.num_players):
            scores_alives.append(self.tiles[self.generals[p]] == p)
        return scores_alives

