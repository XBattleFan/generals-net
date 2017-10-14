
import logging
import json
import threading
import time
from websocket import create_connection, WebSocketConnectionClosedException


_ENDPOINTS = {
    'na': "ws://ws.generals.io/socket.io/?EIO=3&transport=websocket",
    'eu': "ws://euws.generals.io/socket.io/?EIO=3&transport=websocket",
    'bot': "ws://botws.generals.io/socket.io/?EIO=3&transport=websocket",
}
_BOT_KEY = "O13f0dijsf"


class Client(object):
    def __init__(self, region, userid, mode, gameid=None, force_start=True):
        logging.debug("Creating connection")
        self._ws = create_connection(_ENDPOINTS[region])
        self._lock = threading.RLock()

        logging.debug("Starting heartbeat thread")
        _spawn(self._start_sending_heartbeat)

        logging.debug("Joining game")

        if mode == "private":
            if gameid is None:
                raise ValueError("Gameid must be provided for private games")
            self._send(["join_private", gameid, userid, _BOT_KEY])

        elif mode == "1v1":
            self._send(["join_1v1", userid, _BOT_KEY])

        elif mode == "team":
            if gameid is None:
                raise ValueError("Gameid must be provided for team games")
            self._send(["join_team", gameid, userid, _BOT_KEY])

        elif mode == "ffa":
            self._send(["play", userid, _BOT_KEY])

        else:
            raise ValueError("Invalid mode")

        self._send(["set_force_start", gameid, force_start, _BOT_KEY])

        self._start_data = {}
        self._map = []
        self._cities = []

    def move(self, start, end, is_half):
        self._send(["attack", start, end, is_half, _BOT_KEY])

    def get_updates(self):
        while True:
            try:
                msg = self._ws.recv()
            except WebSocketConnectionClosedException:
                break

            if not msg.strip():
                break

            # ignore heartbeats and connection acks
            if msg in {"3", "40"}:
                continue

            # remove numeric prefix
            while msg and msg[0].isdigit():
                msg = msg[1:]

            msg = json.loads(msg)
            if not isinstance(msg, list):
                continue

            if msg[0] == "error_user_id":
                raise ValueError("Already in game")
            elif msg[0] == "game_start":
                logging.info("Game info: {}".format(msg[1]))
                self._start_data = msg[1]
            elif msg[0] == "game_update":
                yield self._make_update(msg[1])
            elif msg[0] in ["game_won", "game_lost"]:
                break
            else:
                logging.info("Unknown message type: {}".format(msg))

    def close(self):
        with self._lock:
            self._ws.close()

    def _make_update(self, data):
        _apply_diff(self._map, data['map_diff'])
        _apply_diff(self._cities, data['cities_diff'])

        player = self._start_data['playerIndex']
        scores = sorted(data['scores'], key=lambda d: d['i'])
        alives = [not d['dead'] for d in scores]

        width, height = self._map[:2]
        n = width * height
        return {
            'done': sum(alives) <= 1,
            'width': width,
            'height': height,
            'generals': data['generals'],
            'cities': self._cities,
            'tiles': self._map[2+n : ],
            'armies': self._map[2 : 2+n],
            'scores_lands': [d['tiles'] for d in scores],
            'scores_units': [d['total'] for d in scores],
            'scores_alives': alives,
            'turn': data['turn'],
            'player': player,

            'usernames': self._start_data['usernames'],
            'teams': self._start_data.get('teams', []),
            'replay': self._start_data['replay_id'],
        }

    def _start_sending_heartbeat(self):
        while True:
            try:
                with self._lock:
                    self._ws.send("2")
            except WebSocketConnectionClosedException:
                break
            time.sleep(10)

    def _send(self, msg):
        try:
            with self._lock:
                self._ws.send("42" + json.dumps(msg))
        except WebSocketConnectionClosedException:
            pass


def _spawn(f):
    t = threading.Thread(target=f)
    t.daemon = True
    t.start()


def _apply_diff(cache, diff):
    i = 0
    a = 0
    while i < len(diff) - 1:

        # offset and length
        a += diff[i]
        n = diff[i+1]

        cache[a:a+n] = diff[i+2:i+2+n]
        a += n
        i += n + 2

    if i == len(diff) - 1:
        cache[:] = cache[:a+diff[i]]
        i += 1

    assert i == len(diff)

