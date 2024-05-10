from network import Network, CRAZY_NEGATIVE_VALUE
from status_loader import StatusLoader
from config import Config

import colorsys
import torch
import numpy as np
import copy
import sys
import math
import argparse
import time
import select

BOARD_SIZE = 19
KOMI = 7.5
INPUT_CHANNELS = 43

SCORING_AREA = 0
SCORING_TERRITORY = 1

MAX_LADDER_NODES = 500
LADDER_INVL = -1
LADDER_NONE = 0
LADDER_DEAD = 1
LADDER_ESCP = 2
LADDER_ATAR = 3
LADDER_TAKE = 4

BLACK = 0
WHITE = 1
EMPTY = 2
INVLD = 3

NUM_VERTICES = (BOARD_SIZE+2) ** 2 # max vertices number
NUM_INTESECTIONS = BOARD_SIZE ** 2 # max intersections number

PASS = -1  # pass
RESIGN = -2 # resign
NULL_VERTEX = NUM_VERTICES+1 # invalid position

def stderr_write(val):
    sys.stderr.write(val)
    sys.stderr.flush()

def stdout_write(val):
    sys.stdout.write(val)
    sys.stdout.flush()

class StoneLiberty(object):
    def __init__(self):
        self.lib_cnt = NULL_VERTEX  # liberty count
        self.v_atr = NULL_VERTEX  # liberty position if in atari
        self.libs = set()  # set of liberty positions

    def clear(self):
        # Reset itself.
        self.lib_cnt = NULL_VERTEX
        self.v_atr = NULL_VERTEX
        self.libs.clear()

    def set(self):
        # Set one stone.
        self.lib_cnt = 0
        self.v_atr = NULL_VERTEX
        self.libs.clear()

    def add(self, v):
        # Add liberty at v.
        if v not in self.libs:
            self.libs.add(v)
            self.lib_cnt += 1
            self.v_atr = v

    def sub(self, v):
        # Remove liberty at v.
        if v in self.libs:
            self.libs.remove(v)
            self.lib_cnt -= 1

    def merge(self, other):
        # Merge itself with another stone.
        self.libs |= other.libs
        self.lib_cnt = len(self.libs)
        if self.lib_cnt == 1:
            for lib in self.libs:
                self.v_atr = lib

'''
 What is the vertex? Vertex is not real board position. It is mail-box position. For example,
 We set the board size to 5. The real board looks like

           a b c d e
        1  . . . . .
        2  . . . . .
        3  . . . . .
        4  . . . . .
        5  . . . . .

 We define the coordinate as index, from a1 to e5. There is some problem to shife the index. The
 shift operation may out of the board. For example, we want to find all positions of adjacent a1
 index. There are two positions out of the board. One way to deal with it is to check out the
 boundary. Another fast way to deal with it is mail-box struct. Here is the mail-box looks like

           a b c d e
         - - - - - - -
       1 - . . . . . -
       2 - . . . . . -
       3 - . . . . . -
       4 - . . . . . -
       5 - . . . . . -
         - - - - - - -

 The board size is changed from 5 to 7. We define the new coordinate as vertex. With mail-box,
 we don't need to waste time to check out the boundary any more. Notice that '-' is out of board
 position.

'''

class Board(object):
    def __init__(self, board_size=BOARD_SIZE, komi=KOMI, scoring_rule=SCORING_AREA):
        self.state = np.full(NUM_VERTICES, INVLD) # positions state
        self.sl = [StoneLiberty() for _ in range(NUM_VERTICES)]  # stone liberties
        self.reset(board_size, komi)
        self.scoring_rule = scoring_rule

    def reset(self, board_size, komi):
        # Initialize all board data with current board size and komi.

        self.board_size = min(board_size, BOARD_SIZE)
        self.num_intersections = self.board_size ** 2
        self.num_vertices = (self.board_size+2) ** 2
        self.komi = komi
        ebsize = board_size+2
        self.dir4 = [1, ebsize, -1, -ebsize]
        self.diag4 = [1 + ebsize, ebsize - 1, -ebsize - 1, 1 - ebsize]

        for vtx in range(self.num_vertices):
            self.state[vtx] = INVLD  # set invalid for out border

        for idx in range(self.num_intersections):
            self.state[self.index_to_vertex(idx)] = EMPTY  # set empty for intersetions

        '''
        self.id, self,next, self.stones are basic data struct for strings. By
        these structs, we can search a whole string more fast. For exmple, we
        have the boards looks like
        
        board position
           a b c d e
        1| . . . . .
        2| . x x x .
        3| . . . . .
        4| . x x . .
        5| . . . . .
        
        vertex position
           a  b  c  d  e
        1| 8  9  10 11 12
        2| 15 16 17 18 19
        3| 22 23 24 25 26
        4| 29 30 31 32 33
        5| 36 37 38 39 40

        self.id
           a  b  c  d  e
        1| .  .  .  .  .
        2| .  16 16 16 .
        3| .  .  .  .  .
        4| .  30 30 .  .
        5| .  .  .  .  .

        self.next
           a  b  c  d  e
        1| .  .  .  .  .
        2| .  17 18 16 .
        3| .  .  .  .  .
        4| .  31 30 .  .
        5| .  .  .  .  .

        self.stones
           a  b  c  d  e
        1| .  .  .  .  .
        2| .  3  .  .  .
        3| .  .  .  .  .
        4| .  2  .  .  .
        5| .  .  .  .  .

        If we want to search the string 16, just simply start from its
        id (the string parent vertex). The pseudo code looks like
        
        start_pos = id[vertex]
        next_pos = start_pos
        {
            next_pos = next[next_pos]
        } while(next_pos != start_pos)

        '''

        self.id = np.arange(NUM_VERTICES)  # the id(parent vertex) of string
        self.next = np.arange(NUM_VERTICES)  # next position in the same string
        self.stones = np.zeros(NUM_VERTICES) # the string size

        for i in range(NUM_VERTICES):
            self.sl[i].clear() # clear liberties

        self.num_passes = 0 # number of passes played.
        self.ko = NULL_VERTEX  # illegal position due to Ko
        self.to_move = BLACK  # black
        self.move_num = 0  # move number
        self.last_move = NULL_VERTEX  # last move
        self.removed_cnt = 0  # removed stones count
        self.history = [] # history board positions.
        self.history_move = [] # history last move.

    def copy(self):
        # Deep copy the board to another board. But they will share the same
        # history board positions.

        b_cpy = Board(self.board_size, self.komi)
        b_cpy.state = np.copy(self.state)
        b_cpy.id = np.copy(self.id)
        b_cpy.next = np.copy(self.next)
        b_cpy.stones = np.copy(self.stones)
        for i in range(NUM_VERTICES):
            b_cpy.sl[i].lib_cnt = self.sl[i].lib_cnt
            b_cpy.sl[i].v_atr = self.sl[i].v_atr
            b_cpy.sl[i].libs |= self.sl[i].libs

        b_cpy.num_passes = self.num_passes
        b_cpy.ko = self.ko
        b_cpy.to_move = self.to_move
        b_cpy.move_num = self.move_num
        b_cpy.last_move = self.last_move
        b_cpy.removed_cnt = self.removed_cnt

        for h in self.history:
            b_cpy.history.append(h)

        for m in self.history_move:
            b_cpy.history_move.append(m)

        return b_cpy

    def _remove(self, v):
        # Remove a string including v.

        v_tmp = v
        removed = 0
        while True:
            removed += 1
            self.state[v_tmp] = EMPTY  # set empty
            self.id[v_tmp] = v_tmp  # reset id
            for d in self.dir4:
                nv = v_tmp + d
                # Add liberty to neighbor strings.
                self.sl[self.id[nv]].add(v_tmp)
            v_next = self.next[v_tmp]
            self.next[v_tmp] = v_tmp
            v_tmp = v_next
            if v_tmp == v:
                break  # Finish when all stones are removed.
        return removed

    def _merge(self, v1, v2):
        '''
        board position
           a  b  c  d  e
        1| .  .  .  .  .
        2| .  x  x  x  .
        3| . [x] .  .  .
        4| .  x  .  .  .
        5| .  .  .  .  .

        Merge two strings...
        
            [before]        >>         [after]

        self.id
           a  b  c  d  e             a  b  c  d  e
        1| .  .  .  .  .          1| .  .  .  .  .
        2| .  16 16 16 .          2| .  16 16 16 .
        3| .  30 .  .  .    >>    3| .  16 .  .  .
        4| .  30 .  .  .          4| .  16 .  .  .
        5| .  .  .  .  .          5| .  .  .  .  .

        self.next
           a  b  c  d  e             a  b  c  d  e
        1| .  .  .  .  .          1| .  .  .  .  .
        2| .  17 18 16 .          2| .  30 18 16 .
        3| .  30 .  .  .    >>    3| .  17 .  .  .
        4| .  23 .  .  .          4| .  23 .  .  .
        5| .  .  .  .  .          5| .  .  .  .  .

        self.stones
           a  b  c  d  e             a  b  c  d  e
        1| .  .  .  .  .          1| .  .  .  .  .
        2| .  3  .  .  .          2| .  5  .  .  .
        3| .  .  .  .  .    >>    3| .  .  .  .  .
        4| .  2  .  .  .          4| .  .  .  .  .
        5| .  .  .  .  .          5| .  .  .  .  .

        '''

        # Merge string including v1 with string including v2.

        id_base = self.id[v1]
        id_add = self.id[v2]

        # We want the large string merges the small string.
        if self.stones[id_base] < self.stones[id_add]:
            id_base, id_add = id_add, id_base  # swap

        self.sl[id_base].merge(self.sl[id_add])
        self.stones[id_base] += self.stones[id_add]

        v_tmp = id_add
        while True:
            self.id[v_tmp] = id_base  # change id to id_base
            v_tmp = self.next[v_tmp]
            if v_tmp == id_add:
                break
        # Swap next id for circulation.
        self.next[v1], self.next[v2] = self.next[v2], self.next[v1]

    def _place_stone(self, v):
        # Play a stone on the board and try to merge itself with adjacent strings.

        # Set one stone to the board and prepare data.
        self.state[v] = self.to_move
        self.id[v] = v
        self.stones[v] = 1
        self.sl[v].set()

        for d in self.dir4:
            nv = v + d
            if self.state[nv] == EMPTY:
                self.sl[self.id[v]].add(nv)  # Add liberty to itself.
            else:
                self.sl[self.id[nv]].sub(v)  # Remove liberty from opponent's string.

        # Merge the stone with my string.
        for d in self.dir4:
            nv = v + d
            if self.state[nv] == self.to_move and self.id[nv] != self.id[v]:
                self._merge(v, nv)

        # Remove the opponent's string.
        self.removed_cnt = 0
        for d in self.dir4:
            nv = v + d
            if self.state[nv] == int(self.to_move == 0) and \
                    self.sl[self.id[nv]].lib_cnt == 0:
                self.removed_cnt += self._remove(nv)

    def legal(self, v):
        # Reture true if the move is legal.

        if v == PASS:
            # The pass move is always legal in any condition.
            return True
        elif v == self.ko or self.state[v] != EMPTY:
            # The move is ko move.
            return False

        stone_cnt = [0, 0]
        atr_cnt = [0, 0] # atari count
        for d in self.dir4:
            nv = v + d
            c = self.state[nv]
            if c == EMPTY:
                return True
            elif c <= 1: # The color must be black or white
                stone_cnt[c] += 1
                if self.sl[self.id[nv]].lib_cnt == 1:
                    atr_cnt[c] += 1

        return (atr_cnt[int(self.to_move == 0)] != 0 or # That means we can eat other stones.
                atr_cnt[self.to_move] < stone_cnt[self.to_move]) # That means we have enough liberty to live.

    def play(self, v):
        # Play the move and update board data if the move is legal.

        if not self.legal(v):
            return False
        else:
            if v == PASS:
                # We should be stop it if the number of passes is bigger than 2.
                # Be sure to check the number of passes before playing it.
                self.num_passes += 1
                self.ko = NULL_VERTEX
            else:
                self._place_stone(v)
                id = self.id[v]
                self.ko = NULL_VERTEX
                if self.removed_cnt == 1 and \
                        self.sl[id].lib_cnt == 1 and \
                        self.stones[id] == 1:
                    # Set the ko move if the last move only captured one and was surround
                    # by opponent's stones.
                    self.ko = self.sl[id].v_atr
                self.num_passes = 0

        self.last_move = v
        self.to_move = int(self.to_move == 0) # switch side
        self.move_num += 1

        # Push the current board positions to history.
        self.history.append(copy.deepcopy(self.state))
        self.history_move.append(self.last_move)

        return True

    def _compute_reach_color(self, color, buf=None):
        # This is simple BFS algorithm to compute evey reachable vertices.

        queue = []
        reachable = 0
        if buf is None:
            buf = [False] * NUM_VERTICES

        # Collect my positions.
        for v in range(self.num_vertices):
            if self.state[v] == color:
                reachable += 1
                buf[v] = True
                queue.append(v)

        # Now start the BFS algorithm to search all reachable positions.
        while len(queue) != 0:
            v = queue.pop(0)
            for d in self.dir4:
                nv = v + d
                if self.state[nv] == EMPTY and buf[nv] == False:
                    reachable += 1
                    queue.append(nv)
                    buf[nv] = True
        return reachable

    def get_scoring_rule(self):
        if self.scoring_rule == SCORING_TERRITORY:
            assert False, "don't support for scoring territorry now"
            return 1.0
        return 0.0 # default, scoring area

    def get_wave(self):
        if self.scoring_rule == SCORING_TERRITORY:
            return 0.0

        curr_komi = self.komi
        if self.to_move == WHITE:
            curr_komi = 0.0 - curr_komi

        is_board_area_even = (self.num_intersections % 2) == 0
        if is_board_area_even:
            komi_floor = math.floor(curr_komi / 2) * 2
        else:
            komi_floor = math.floor((curr_komi - 1) / 2) * 2 + 1

        delta =  curr_komi - komi_floor;
        delta = max(delta, 0.);
        delta = min(delta, 2.);

        if delta < 0.5:
            wave = delta
        elif delta < 1.5:
            wave = 1. - delta
        else:
            wave = delta - 2.
        return wave;

    def _get_owner_map(self):
        ownermap = np.full(NUM_INTESECTIONS, INVLD)
        black_buf = [False] * NUM_VERTICES
        white_buf = [False] * NUM_VERTICES
        self._compute_reach_color(BLACK, black_buf)
        self._compute_reach_color(WHITE, white_buf)

        for v in range(self.num_vertices):
            i = self.vertex_to_index(v)
            if black_buf[v] and white_buf[v]:
                ownermap[i] = EMPTY
            elif black_buf[v]:
                ownermap[i] = BLACK
            elif white_buf[v]:
                ownermap[i] = WHITE
        return ownermap

    def _beason_search(self, beasonmap, c):
        pass

    def _get_beason_map(self):
        beasonmap  = np.full(NUM_INTESECTIONS, INVLD)
        vitals = np.full(NUM_VERTICES, False)

        self._beason_search(beasonmap, BLACK)
        self._beason_search(beasonmap, WHITE)
        return beasonmap

    def _ladder_search(self, v, n):
        n += 1
        if n >= MAX_LADDER_NODES:
            return False, n 

        prey_c = self.state[v]
        hunter_c = int(prey_c == 0)
        v_tmp = v
        while True:
            for d in self.dir4:
                nv = v_tmp + d
                if self.state[nv] == hunter_c and \
                   self.sl[self.id[nv]].lib_cnt == 1:
                   # Prey can capture surround strings. Simply
                   # think it is not ladder.
                   return False, n
            v_tmp = self.next[v_tmp]
            if v_tmp == v:
                break

        prey_move = next(iter(self.sl[self.id[v]].libs))
        if not self.play(prey_move): # prey play
            return True, n # The prey string can not escape...
        lib_cnt = self.sl[self.id[v]].lib_cnt

        if lib_cnt == 1:
            # The prey string is dead. It is ladder.
            return True, n
        elif lib_cnt == 2:
            res = False
            candidate = list(self.sl[self.id[v]].libs)

            for lv in self.sl[self.id[v]].libs:
                virtual_libs = 0
                for d in self.dir4:
                     if self.state[lv + d] == EMPTY:
                         virtual_libs += 1
                if virtual_libs > 2:
                    # Prey will get to many libs. Hunter must play
                    # here.
                    candidate = [lv]
                    break

            for lv in candidate:
                b_ladder = self.copy()
                if b_ladder.play(lv): # hunter plays
                    if b_ladder.sl[b_ladder.id[lv]].lib_cnt != 1: # not self-atari
                        res, n = b_ladder._ladder_search(v, n)
                        if res:
                            break
            return res, n
        # Too many libs, it is not ladder.
        return False, n

    def _is_ladder(self, v):
        if self.state[v] == EMPTY or \
               self.state[v] == INVLD:
            return False, []

        lib_cnt = self.sl[self.id[v]].lib_cnt
        c = self.state[v]

        vital_moves = []
        if lib_cnt == 1:
            b_ladder = self.copy()
            b_ladder.to_move = c # prey color
            res, _ = b_ladder._ladder_search(v, 0)
            if res:
                prey_move = next(iter(self.sl[self.id[v]].libs))
                vital_moves.append(prey_move)
        elif lib_cnt == 2:
            n = 0
            for lv in self.sl[self.id[v]].libs:
                b_ladder = self.copy()
                b_ladder.to_move = int(c == 0) # hunter color
                if b_ladder.play(lv): # hunter play
                    if b_ladder.sl[b_ladder.id[lv]].lib_cnt != 1: # not self-atari
                        res, n = b_ladder._ladder_search(v, n)
                        if res:
                            vital_moves.append(lv)
        return len(vital_moves) != 0, vital_moves

    def _get_ladder_map(self):
        laddermap = np.full(NUM_VERTICES, LADDER_INVL)

        for v in range(self.num_vertices):
            if self.state[v] == EMPTY:
                laddermap[v] = LADDER_NONE

        for v in range(self.num_vertices):
            if self.state[v] == INVLD or \
                   self.state[v] == EMPTY or \
                   laddermap[v] == LADDER_NONE:
                continue

            ladder, vital_moves = self._is_ladder(v)
            lib_cnt = self.sl[self.id[v]].lib_cnt

            if ladder:
                for vm in vital_moves:
                    if lib_cnt == 1:
                        laddermap[vm] = LADDER_TAKE
                    elif lib_cnt == 2:
                        laddermap[vm] = LADDER_ATAR
            v_tmp = v
            while True:
                if not ladder:
                    laddermap[v_tmp] = LADDER_NONE
                else:
                    if lib_cnt == 1:
                        laddermap[v_tmp] = LADDER_DEAD
                    elif lib_cnt == 2:
                        laddermap[v_tmp] = LADDER_ESCP
                v_tmp = self.next[v_tmp]
                if v_tmp == v:
                    break
        return laddermap

    def _remove(self, v):
        # Remove a string including v.

        v_tmp = v
        removed = 0
        while True:
            removed += 1
            self.state[v_tmp] = EMPTY  # set empty
            self.id[v_tmp] = v_tmp  # reset id
            for d in self.dir4:
                nv = v_tmp + d
                # Add liberty to neighbor strings.
                self.sl[self.id[nv]].add(v_tmp)
            v_next = self.next[v_tmp]
            self.next[v_tmp] = v_tmp
            v_tmp = v_next
            if v_tmp == v:
                break  # Finish when all stones are removed.
        return removed

    def final_score(self):
        # Scored the board area with Tromp-Taylor rule.
        return self._compute_reach_color(BLACK) - self._compute_reach_color(WHITE) - self.komi

    def get_x(self, v):
        # vertex to x
        return v % (self.board_size+2) - 1

    def get_y(self, v):
        # vertex to y
        return v // (self.board_size+2) - 1

    def get_vertex(self, x, y):
        # x, y to vertex
        return (y+1) * (self.board_size+2) + (x+1)
        
    def get_index(self, x, y):
        # x, y to index
        return y * self.board_size + x

    def vertex_to_index(self, v):
        # vertex to index
        return self.get_index(self.get_x(v), self.get_y(v))
        
    def index_to_vertex(self, idx):
        # index to vertex
        return self.get_vertex(idx % self.board_size, idx // self.board_size)

    def vertex_to_text(self, vtx):
        # vertex to GTP move

        if vtx == PASS:
            return "pass"
        elif vtx == RESIGN:
            return "resign"
        
        x = self.get_x(vtx)
        y = self.get_y(vtx)
        offset = 1 if x >= 8 else 0 # skip 'I'
        return "".join([chr(x + ord('A') + offset), str(y+1)])

    def vertex_to_feature_index(self, v):
        x = self.get_x(v)
        y = self.get_y(v)
        return y * BOARD_SIZE + x

    def get_features(self):
        # planes 1 -24 : last 8 history moves
        # plane     25 : ko move
        # plane  26-29 : pass-alive and pass-dead area
        # planes 30-33 : strings with 1, 2, 3 and 4 liberties 
        # planes 34-37 : ladder features
        # plane     38 : rule
        # plane     39 : wave
        # plane     40 : komi/20
        # plane     41 : -komi/20
        # plane     42 : intersections/361
        # plane     43 : fill ones
        my_color = self.to_move
        opp_color = (self.to_move + 1) % 2
        past_moves = min(8, len(self.history))
        
        features = np.zeros((INPUT_CHANNELS, NUM_INTESECTIONS), dtype=np.float32)

        # planes 1-24
        for p in range(past_moves):
            i = len(self.history) - p - 1

            # Fill past board positions features.
            h = self.history[i]
            for v in range(self.num_vertices):
                c = h[v]
                if c == my_color:
                    features[p*3+0, self.vertex_to_feature_index(v)] = 1
                elif c == opp_color:
                    features[p*3+1, self.vertex_to_feature_index(v)] = 1

            m = self.history_move[i]
            if m != PASS and m != RESIGN and m != NULL_VERTEX:
                features[p*3+2, self.vertex_to_feature_index(m)] = 1

        # planes 25
        if self.ko != NULL_VERTEX:
            features[24, self.vertex_to_feature_index(self.ko)] = 1

        # planes 26-29
        ownermap = self._get_owner_map()
        beasonmap = self._get_beason_map()
        for v in range(self.num_vertices):
            if self.state[v] != INVLD:
                if beasonmap[self.vertex_to_index(v)] == my_color:
                    features[25, self.vertex_to_feature_index(v)] = 1
                elif beasonmap[self.vertex_to_index(v)] == opp_color:
                    features[26, self.vertex_to_feature_index(v)] = 1
                if ownermap[self.vertex_to_index(v)] == my_color:
                    features[27, self.vertex_to_feature_index(v)] = 1
                elif ownermap[self.vertex_to_index(v)] == opp_color:
                    features[28, self.vertex_to_feature_index(v)] = 1

        # planes 30-33
        for i in range(4):
            for v in range(self.num_vertices):
                c = self.state[v]
                if (c == BLACK or c == WHITE) and \
                       self.sl[self.id[v]].lib_cnt == i+1:
                    features[29+i, self.vertex_to_feature_index(v)] = 1

        # planes 34-37
        laddermap = self._get_ladder_map()
        for v in range(self.num_vertices):
            if self.state[v] != INVLD:
                if laddermap[v] == LADDER_DEAD:
                    features[33, self.vertex_to_feature_index(v)] = 1
                elif laddermap[v] == LADDER_ESCP:
                    features[34, self.vertex_to_feature_index(v)] = 1
                elif laddermap[v] == LADDER_ATAR:
                    features[35, self.vertex_to_feature_index(v)] = 1
                elif laddermap[v] == LADDER_TAKE:
                    features[36, self.vertex_to_feature_index(v)] = 1

        # planes 38-43
        side_komi = self.komi if my_color == BLACK else -self.komi
        wave = self.get_wave()
        scoring = self.get_scoring_rule()
        for v in range(self.num_vertices):
            if self.state[v] != INVLD:
                features[37, self.vertex_to_feature_index(v)] = scoring
                features[38, self.vertex_to_feature_index(v)] = wave
                features[39, self.vertex_to_feature_index(v)] = side_komi/20.
                features[40, self.vertex_to_feature_index(v)] = -side_komi/20.
                features[41, self.vertex_to_feature_index(v)] = self.num_intersections/361.
                features[42, self.vertex_to_feature_index(v)] = 1.

        return np.reshape(features, (INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE))

    def superko(self):
        # Return true if the current position is superko.

        curr_hash = hash(self.state.tobytes())
        s = len(self.history)
        for p in range(s-1):
            h = self.history[p]
            if hash(h.tobytes()) == curr_hash:
                return True
        return False

    def __str__(self):
        def get_xlabel(bsize):
            X_LABELS = "ABCDEFGHJKLMNOPQRST"
            line_str = "  "
            for x in range(bsize):
                line_str += " " + X_LABELS[x] + " "
            return line_str + "\n"
        out = str()
        out += get_xlabel(self.board_size)

        for y in range(0, self.board_size)[::-1]:  # 9, 8, ..., 1
            line_str = str(y+1) if y >= 9 else " " + str(y+1)
            for x in range(0, self.board_size):
                v = self.get_vertex(x, y)
                x_str = " . "
                color = self.state[v]
                if color <= 1:
                    stone_str = "O" if color == WHITE else "X"
                    if v == self.last_move:
                        x_str = "[" + stone_str + "]"
                    else:
                        x_str = " " + stone_str + " "
                line_str += x_str
            line_str += str(y+1) if y >= 9 else " " + str(y+1)
            out += (line_str + "\n")
        out += get_xlabel(self.board_size)
        out += "komi: {:.2f}".format(self.komi)
        return out + "\n"

class TimeControl:
    def __init__(self):
        self.main_time = 0
        self.byo_time = 7 * 24 * 60 * 60 # one week per move
        self.byo_stones = 1

        self.maintime_left = [0, 0]
        self.byotime_left = [0, 0]
        self.stones_left = [0, 0]
        self.in_byo = [False, False]
        
        self.clock_time = time.time()
        self.reset()

    def check_in_byo(self):
        self.in_byo[0] = True if self.maintime_left[0] <= 0 else False
        self.in_byo[1] = True if self.maintime_left[1] <= 0 else False

    def reset(self):
        self.maintime_left = [self.main_time] * 2
        self.byotime_left = [self.byo_time] * 2
        self.stones_left = [self.byo_stones] * 2
        self.check_in_byo()

    def time_settings(self, main_time, byo_time, byo_stones):
        self.main_time = main_time
        self.byo_time = byo_time
        self.byo_stones = byo_stones
        self.reset()

    def time_left(self, color, time, stones):
        if stones == 0:
            self.maintime_left[color] = time
        else:
            self.maintime_left[color] = 0
            self.byotime_left[color] = time
            self.stones_left[color] = stones
        self.check_in_byo()

    def clock(self):
        self.clock_time = time.time()

    def took_time(self, color):
        remaining_took_time = time.time() - self.clock_time
        if not self.in_byo[color]:
            if self.maintime_left[color] > remaining_took_time:
                self.maintime_left[color] -= remaining_took_time
                remaining_took_time = -1
            else:
                remaining_took_time -= self.maintime_left[color]
                self.maintime_left[color] = 0
                self.in_byo[color] = True

        if self.in_byo[color] and remaining_took_time > 0:
            self.byotime_left[color] -= remaining_took_time
            self.stones_left[color] -= 1
            if self.stones_left[color] == 0:
                self.stones_left[color] = self.byo_stones
                self.byotime_left[color] = self.byo_time

    def get_thinking_time(self, color, board_size, move_num):
        estimate_moves_left = max(4, int(board_size * board_size * 0.4) - move_num)
        lag_buffer = 1 # Remaining some time for network hiccups or GUI lag 
        remaining_time = self.maintime_left[color] + self.byotime_left[color] - lag_buffer
        if self.byo_stones == 0:
            return remaining_time / estimate_moves_left
        return remaining_time / self.stones_left[color]

    def should_stop(self, max_time):
        elapsed = time.time() - self.clock_time
        return elapsed > max_time

    def get_timeleft_string(self, color):
        out = str()
        if not self.in_byo[color]:
            out += "{s} sec".format(
                                 s=int(self.maintime_left[color]))
        else:
            out += "{s} sec, {c} stones".format(
                                             s=int(self.byotime_left[color]),
                                             c=self.stones_left[color])
        return out

    def __str__(self):
        return "".join(["Black: ",
                          self.get_timeleft_string(0),
                           " | White: ",
                           self.get_timeleft_string(1)])

class Node:
    # TODO: Tnue C_PUCT parameter.
    C_PUCT = 0.5 # The PUCT hyperparameter.
    def __init__(self, p):
        self.policy = p  # The network raw policy from its parents node.
        self.nn_eval = 0 # The network raw eval from this node.

        self.values = 0 # The accumulate winrate.
        self.visits = 0 # The accumulate node visits.
                        # The Q value must be equal to (self.values / self.visits)
        self.children = dict() # Next node.

    def inverse(self, v):
        # Swap the side to move winrate. 
        return 1 - v;

    @torch.no_grad()
    def _get_network_result(self, board, network, device):
        def np_softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        planes = torch.from_numpy(board.get_features()).float().to(device)
        pred, _ = network.forward(torch.unsqueeze(planes, 0))

        # Use optimistic policy.
        _, _, _, _, prob, _, wdl, _, scores, _ = pred

        # winrate
        wdl = wdl[0].cpu().detach().numpy()
        wdl = np_softmax(wdl)
        winrate = (wdl[0] - wdl[2] + 1) / 2

        # score
        scores = scores[0].cpu().detach().numpy()
        score = scores[0]

        # prob
        prob = prob[0].cpu().detach().numpy()
        prob = get_valid_spat(prob, board)
        prob = np_softmax(prob)
        return prob, wdl, score

    def expand_children(self, board, network, device):
        if board.last_move == PASS:
            score = board.final_score()
            if (board.to_move == BLACK and score > 0) or \
                    (board.to_move == WHITE and score < 0):
                # Play pass move if we win the game.
                self.children[PASS] = Node(1.0)
                return 1;

        # Compute the net results.
        policy, value, _ = self._get_network_result(board, network, device)

        for idx in range(board.num_intersections):
            vtx = board.index_to_vertex(idx)

            # Remove the all illegal move.
            if board.legal(vtx):
                p = policy[idx]
                self.children[vtx] = Node(p)

        # The pass move is alwaly the legal move. We don't need to
        # check it.
        self.children[PASS] = Node(policy[-1])

        # The nn eval is side-to-move winrate. 
        self.nn_eval = value[0]

        return self.nn_eval

    def remove_superko(self, board):
        # Remove all superko moves.

        remove_list = list()
        for vtx, _ in self.children.items():
            if vtx != PASS:
                next_board = board.copy()
                next_board.play(vtx)
                if next_board.superko():
                    remove_list.append(vtx)
        for vtx in remove_list:
            self.children.pop(vtx)

    def puct_select(self):
        parent_visits = max(self.visits, 1) # The parent visits must great than 1 because we want to get the
                                            # best policy value if it is the first selection.
        numerator = math.sqrt(parent_visits)
        puct_list = list()

        # Select the best node by PUCT algorithm. 
        for vtx, child in self.children.items():
            q_value = 0

            if child.visits != 0:
                q_value = self.inverse(child.values / child.visits)

            puct = q_value + self.C_PUCT * child.policy * (numerator / (1+child.visits))
            puct_list.append((puct, vtx))
        return max(puct_list)[1]

    def update(self, v):
        self.values += v
        self.visits += 1

    def get_best_prob_move(self):
        gather_list = list()
        for vtx, child in self.children.items():
            gather_list.append((child.policy, vtx))
        return max(gather_list)[1]

    def get_best_move(self, resign_threshold):
        # Return best probability move if there are no playouts.
        if self.visits == 1 and \
               resign_threshold is not None:
            if self.values < resign_threshold:
                return 0;
            else:
                return self.get_best_prob_move()

        # Get best move by number of node visits.
        gather_list = list()
        for vtx, child in self.children.items():
            gather_list.append((child.visits, vtx))

        vtx = max(gather_list)[1]
        child = self.children[vtx]

        # Play resin move if we think we have already lost.
        if resign_threshold is not None and \
               self.inverse(child.values / child.visits) < resign_threshold:
            return RESIGN
        return vtx

    def to_string(self, board):
        # Collect some node information in order to debug.

        out = str()
        out += "Root -> W: {:5.2f}%, V: {}\n".format(
                    100.0 * self.values/self.visits,
                    self.visits)

        gather_list = list()
        for vtx, child in self.children.items():
            gather_list.append((child.visits, vtx))
        gather_list.sort(reverse=True)

        for _, vtx in gather_list:
            child = self.children[vtx]
            if child.visits != 0:
                out += "  {:4} -> W: {:5.2f}%, P: {:5.2f}%, V: {}\n".format(
                           board.vertex_to_text(vtx),
                           100.0 * self.inverse(child.values/child.visits),
                           100.0 * child.policy,
                           child.visits)
        return out

    def get_pv(self, board, pv_str):
        # Get the best Principal Variation list since this
        # node.
        if len(self.children) == 0: 
            return pv_str

        next_vtx = self.get_best_move(None)
        next = self.children[next_vtx]
        pv_str += "{} ".format(board.vertex_to_text(next_vtx))
        return next.get_pv(board, pv_str)

    def to_lz_analysis(self, board):
        # Output the leela zero analysis string. Watch the detail
        # here: https://github.com/SabakiHQ/Sabaki/blob/master/docs/guides/engine-analysis-integration.md
        out = str()

        gather_list = list()
        for vtx, child in self.children.items():
            gather_list.append((child.visits, vtx))
        gather_list.sort(reverse=True)

        i = 0
        for _, vtx in gather_list:
            child = self.children[vtx]
            if child.visits != 0:
                winrate = self.inverse(child.values/child.visits)
                prior = child.policy
                lcb = winrate
                order = i
                pv = "{} ".format(board.vertex_to_text(vtx))
                out += "info move {} visits {} winrate {} prior {} lcb {} order {} pv {}".format(
                           board.vertex_to_text(vtx),
                           child.visits,
                           int(10000 * winrate),
                           int(10000 * prior),
                           int(10000 * lcb),
                           order,
                           child.get_pv(board, pv))
                i+=1
        out += '\n'
        return out

class Search:
    def __init__(self, board, network, time_control, device):
        self.root_board = board # Root board positions, all simulation boards will fork from it.
        self.root_node = None # Root node, start the PUCT search from it.
        self.network = network
        self.time_control = time_control
        self.device = device
        self.analysis_tag = {
            "interval" : -1
        }

    def _prepare_root_node(self):
        # Expand the root node first.
        self.root_node = Node(1)
        val = self.root_node.expand_children(self.root_board, self.network, self.device)

        # In order to avoid overhead, we only remove the superko positions in
        # the root.
        self.root_node.remove_superko(self.root_board)
        self.root_node.update(val)

    def _descend(self, color, curr_board, node):
        value = None
        if curr_board.num_passes >= 2:
            # The game is over. Compute the final score.
            score = curr_board.final_score()
            if score > 1e-4:
                # The black player is winner.
                value = 1 if color is BLACK else 0
            elif score < -1e-4:
                # The white player is winner.
                value = 1 if color is WHITE else 0
            else:
                # The game is draw
                value = 0.5
        elif len(node.children) != 0:
            # Select the next node by PUCT algorithm. 
            vtx = node.puct_select()
            curr_board.to_move = color
            curr_board.play(vtx)
            color = (color + 1) % 2
            next_node = node.children[vtx]

            # go to the next node.
            value = self._descend(color, curr_board, next_node)
        else:
            # This is the termainated node. Now try to expand it. 
            value = node.expand_children(curr_board, self.network, self.device)

        assert value != None, ""
        node.update(value)

        return node.inverse(value)

    def ponder(self, playouts, verbose):
        if self.root_board.num_passes >= 2:
            return str()

        analysis_clock = time.time()
        interval = self.analysis_tag["interval"]

        # Try to expand the root node first.
        self._prepare_root_node()

        for p in range(playouts):
            if p != 0 and \
                   interval > 0 and \
                   time.time() - analysis_clock  > interval:
                analysis_clock = time.time()
                stdout_write(self.root_node.to_lz_analysis(self.root_board))

            rlist, _, _ = select.select([sys.stdin], [], [], 0)
            if rlist:
                break

            # Copy the root board because we need to simulate the current board.
            curr_board = self.root_board.copy()
            color = curr_board.to_move

            # Start the Monte Carlo tree search.
            self._descend(color, curr_board, self.root_node)

        out_verbose = self.root_node.to_string(self.root_board)
        if verbose:
            # Dump verbose to stderr because we want to debug it on GTP
            # interface(sabaki).
            stderr_write(out_verbose + "\n")

        return out_verbose

    def think(self, playouts, resign_threshold, verbose):
        # Get the best move with Monte carlo tree. The time controller and max playouts limit
        # the search. More thinking time or playouts is stronger.

        if self.root_board.num_passes >= 2:
            return PASS, str()

        analysis_clock = time.time()
        interval = self.analysis_tag["interval"]
        self.time_control.clock()
        if verbose:
            stderr_write(str(self.time_control) + "\n")

        # Prepare some basic information.
        to_move = self.root_board.to_move
        bsize = self.root_board.board_size
        move_num = self.root_board.move_num

        # Compute thinking time limit.
        max_time = self.time_control.get_thinking_time(to_move, bsize, move_num)

        # Try to expand the root node first.
        self._prepare_root_node()

        for p in range(playouts):
            if p != 0 and \
                   interval > 0 and \
                   time.time() - analysis_clock  > interval:
                analysis_clock = time.time()
                stdout_write(self.root_node.to_lz_analysis(self.root_board))

            if self.time_control.should_stop(max_time):
                break

            # Copy the root board because we need to simulate the current board.
            curr_board = self.root_board.copy()
            color = curr_board.to_move

            # Start the Monte Carlo tree search.
            self._descend(color, curr_board, self.root_node)

        # Eat the remaining time. 
        self.time_control.took_time(to_move)

        out_verbose = self.root_node.to_string(self.root_board)
        if verbose:
            # Dump verbose to stderr because we want to debug it on GTP
            # interface(sabaki).
            stderr_write(out_verbose)
            stderr_write(str(self.time_control))
            stderr_write("\n")

        return self.root_node.get_best_move(resign_threshold), out_verbose

def load_checkpoint(json_path, checkpoint, use_swa):
    cfg = None
    if json_path is None and not checkpoint is None:
        loader = StatusLoader()
        loader.load(checkpoint)
        cfg = Config(loader.get_json_str(), False)
    else:
        cfg = Config(json_path)

    if cfg is None:
        raise Exception("The config file does not exist.")

    cfg.boardsize = BOARD_SIZE
    net = Network(cfg)
    stderr_write("Load the weights from: {}\n".format(checkpoint))
    stderr_write(net.simple_info())

    if not checkpoint is None:
        loader = StatusLoader()
        loader.load(checkpoint)
        loader.load_model(net)
        if use_swa:
            loader.load_swa_model(net)
    net.eval()
    return net

def get_valid_spat(net_pred, board):
    valid_size = board.num_intersections
    if net_pred.size == NUM_INTESECTIONS + 1:
        valid_size += 1

    valid_spat = np.zeros(valid_size)

    for y in range(board.board_size):
        for x in range(board.board_size):
            valid_idx = board.get_index(x, y)
            net_idx = y * BOARD_SIZE + x
            valid_spat[valid_idx] = net_pred[net_idx]

    if net_pred.size == NUM_INTESECTIONS + 1:
        valid_spat[-1] = net_pred[-1]

    return valid_spat

def get_move_text(x, y):
    return "{}{}".format("ABCDEFGHJKLMNOPQRST"[x], y+1)

def gogui_policy_order(prob, board):
    def np_softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    prob = prob[0].cpu().detach().numpy()
    prob = get_valid_spat(prob, board)
    prob = np_softmax(prob)

    ordered_list = list()
    for idx in range(board.num_intersections):
        ordered_list.append((prob[idx], idx))

    out = str()
    ordered_list.sort(reverse=True, key=lambda x: x[0])
    for order in range(12):
        _, idx = ordered_list[order]
        vtx = board.index_to_vertex(idx)
        x = board.get_x(vtx)
        y = board.get_y(vtx)
        out += "LABEL {} {}\n".format(get_move_text(x, y), order+1)
    out = out[:-1]
    return out

def gogui_policy_rating(prob, board):
    def np_softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    prob = prob[0].cpu().detach().numpy()
    prob = get_valid_spat(prob, board)
    prob = np_softmax(prob)

    out = str()
    for y in range(board.board_size):
        for x in range(board.board_size):
            val = prob[board.get_index(x, y)]
            if val > 1./board.num_intersections:
                out += "LABEL {} {}\n".format(get_move_text(x, y), round(100. * val))
    out = out[:-1]
    return out

def gogui_policy_heatmap(prob, board):
    def np_softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def value_to_code(val):
        def hsv2rgb(h,s,v):
            return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

        h1, h2 = 145, 215
        w = h2 - h1
        w2 = 20

        h = (1.0 - val) * (242 - w + w2)
        s = 1.0
        v = 1.0

        if (h1 <= h and h <= h1 + w2):
            h = h1 + (h - h1) * w / w2
            m = w / 2
            v -= (m - abs(h - (h1 + m))) * 0.2 / m
        elif h >= h1 + w2:
            h += w - w2

        h0 = 100
        m0 = (h2 - h0) / 2
        if h0 <= h and h <= h2:
            v -= (m0 - abs(h - (h0 + m0))) * 0.2 / m0
        r, g, b = hsv2rgb(h/360, s, v)
        return "#{0:02x}{1:02x}{2:02x}".format(r, g, b)

    prob = prob[0].cpu().detach().numpy()
    prob = get_valid_spat(prob, board)
    prob = np_softmax(prob)

    out = str()
    for y in range(board.board_size):
        for x in range(board.board_size):
            val = prob[board.get_index(x, y)]
            if val > 0.0001:
                val = math.sqrt(val) # highlight
            out += "COLOR {} {}\n".format(value_to_code(val), get_move_text(x, y))
    out = out[:-1]
    return out

def gogui_ownership_heatmap(ownership, board):
    def value_to_code(val, inves):
        def hsv2rgb(h,s,v):
            return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))
        h = 0.
        s = 0.
        v = val if not inves else 1. - val
        r, g, b = hsv2rgb(h/360, s, v)
        return "#{0:02x}{1:02x}{2:02x}".format(r, g, b)

    ownership = ownership[0].cpu().detach().numpy()
    ownership = get_valid_spat(ownership, board)
    inves = board.to_move == BLACK

    out = str()
    for y in range(board.board_size):
        for x in range(board.board_size):
            val = (ownership[board.get_index(x, y)] + 1.) / 2.0
            out += "COLOR {} {}\n".format(value_to_code(val, inves), get_move_text(x, y))
    out = out[:-1]
    return out

def gogui_ownership_influence(ownership, board):
    ownership = ownership[0].cpu().detach().numpy()
    ownership = get_valid_spat(ownership, board)
    inves = board.to_move == WHITE

    out = "INFLUENCE"
    for y in range(board.board_size):
        for x in range(board.board_size):
            val = ownership[board.get_index(x, y)]
            if inves:
                val = -val
            out += " {} {:.2f}".format(get_move_text(x, y), val)
    return out

def gogui_ladder_heatmap(laddermap, board):
    def value_to_code(val):
        def hsv2rgb(h,s,v):
            return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

        h1, h2 = 145, 215
        w = h2 - h1
        w2 = 20

        h = (1.0 - val) * (242 - w + w2)
        s = 1.0
        v = 1.0

        if (h1 <= h and h <= h1 + w2):
            h = h1 + (h - h1) * w / w2
            m = w / 2
            v -= (m - abs(h - (h1 + m))) * 0.2 / m
        elif h >= h1 + w2:
            h += w - w2

        h0 = 100
        m0 = (h2 - h0) / 2
        if h0 <= h and h <= h2:
            v -= (m0 - abs(h - (h0 + m0))) * 0.2 / m0
        r, g, b = hsv2rgb(h/360, s, v)
        return "#{0:02x}{1:02x}{2:02x}".format(r, g, b)

    out = str()
    for y in range(board.board_size):
        for x in range(board.board_size): 
            ladder = laddermap[board.get_vertex(x, y)]
            if ladder == LADDER_ATAR:
                val = 0.2
            elif ladder == LADDER_TAKE:
                val = 0.4
            elif ladder == LADDER_ESCP:
                val = 0.8
            elif ladder == LADDER_DEAD:
                val = 1.0
            else:
                val = 0.0
            out += "COLOR {} {}\n".format(value_to_code(val), get_move_text(x, y))
    out = out[:-1]
    return out

def print_winrate(wdl, scores):
    def np_softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    wdl = wdl[0].cpu().detach().numpy()
    wdl = np_softmax(wdl)
    scores = scores[0].cpu().detach().numpy()
    score = scores[0]

    winrate = (wdl[0] - wdl[2] + 1) / 2
    stderr_write("winrate= {:.2f}%\n".format(100 * winrate))
    stderr_write("score= {:.2f}\n".format(score))

def print_raw_nn(pred, board):
    def np_softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    prob, _, _, _, _, ownership, wdl, _, scores, _ = pred

    # prob
    prob = prob[0].cpu().detach().numpy()
    prob = get_valid_spat(prob, board)
    prob = np_softmax(prob)

    # winrate
    wdl = wdl[0].cpu().detach().numpy()
    wdl = np_softmax(wdl)
    winrate = (wdl[0] - wdl[2] + 1) / 2

    # score
    scores = scores[0].cpu().detach().numpy()
    score = scores[0]

    # ownership
    ownership = ownership[0].cpu().detach().numpy()
    ownership = get_valid_spat(ownership, board)

    stderr_write("winrate= {:.6f}%\n".format(100 * winrate))
    stderr_write("score= {:.6f}\n".format(score))

    prob_out = str()
    for y in range(board.board_size):
        for x in range(board.board_size): 
            idx = (board.board_size - y - 1) * board.board_size + x
            prob_out += "  {:.6f}".format(prob[idx])
        prob_out += "\n"
    stderr_write("policy=\n{}".format(prob_out))
    stderr_write("pass policy= {:.6f}\n".format(prob[-1]))

    ownership_out = str()
    for y in range(board.board_size):
        for x in range(board.board_size): 
            idx = (board.board_size - y - 1) * board.board_size + x
            val = ownership[idx]
            if val < 0:
                ownership_out += " {:.6f}".format(val)
            else:
                ownership_out += "  {:.6f}".format(val)
        ownership_out += "\n"
    stderr_write("ownership=\n{}".format(ownership_out))

def print_planes(board):
    planes = board.get_features()

    planes_out = str()
    for p in range(INPUT_CHANNELS):
        planes_out += "planes: {}\n".format(p+1)
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                val = planes[p, BOARD_SIZE - y - 1, x]
                if val < 0:
                    planes_out += " {:.2f}".format(val)
                else:
                    planes_out += "  {:.2f}".format(val)
            planes_out += "\n"
        planes_out += "\n"
    stderr_write("planes=\n{}".format(planes_out))

def gtp_loop(args):
    def gtp_print(res, success=True):
        if success:
            stdout_write("= {}\n\n".format(res))
        else:
            stdout_write("? {}\n\n".format(res))

    board = Board(BOARD_SIZE, KOMI, SCORING_AREA)
    net = load_checkpoint(args.json, args.checkpoint, args.use_swa)

    use_gpu = args.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda") if use_gpu else torch.device("cpu")
    net = net.to(device)

    time_control = TimeControl()

    if use_gpu:
        stderr_write("Enable the GPU device...\n")

    while True:
        inputs = sys.stdin.readline().strip().split()
        if len(inputs) == 0:
            continue

        main = inputs[0]

        if main == "quit":
            gtp_print("")
            break
        elif main == "name":
            gtp_print("pysayuri")
        elif main == "version":
            gtp_print("0.1.0")
        elif main == "protocol_version":
            gtp_print("2")
        elif main == "list_commands":
            supported_list = [
                "name",
                "version",
                "protocol_version",
                "list_commands",
                "showboard",
                "clear_board",
                "boardsize",
                "komi",
                "play",
                "genmove",
                "time_settings",
                "time_left",
                "raw-nn",
                "planes",
                "save_bin_weights",
                "gogui-analyze_commands",
                "lz-genmove_analyze",
                "lz-analyze"
            ]
            out = str()
            for i in supported_list:
                out += (i + "\n")
            out = out[:-1]
            gtp_print(out)
        elif main == "showboard":
            gtp_print("\n" + str(board))
        elif main == "clear_board":
            board.reset(board.board_size, board.komi)
            gtp_print("")
        elif main == "boardsize":
            board.reset(int(inputs[1]), board.komi)
            gtp_print("")
        elif main == "komi":
            board.komi = float(inputs[1])
            gtp_print("")
        elif main == "play":
            color, move = inputs[1].lower(), inputs[2].lower()
            c, vtx = INVLD, None
            if color[:1] == "b":
                c = BLACK
            elif color[:1] == "w":
                c = WHITE

            if move == "pass":
                vtx = PASS
            elif move == "resign":
                vtx = RESIGN
            else:
                x = ord(move[0]) - (ord('A') if ord(move[0]) < ord('a') else ord('a'))
                y = int(move[1:]) - 1
                if x >= 8:
                    x -= 1
                vtx = board.get_vertex(x,y)
            if c != INVLD and vtx is not None:
                board.to_move = c
                board.play(vtx)
            gtp_print("")
        elif main == "genmove":
            color = inputs[1]
            c = INVLD
            if color.lower()[:1] == "b":
                c = BLACK
            elif color.lower()[:1] == "w":
                c = WHITE
            board.to_move = c

            search = Search(board, net, time_control, device)
            vtx, _ = search.think(args.playouts, args.resign_threshold, args.verbose)
            board.play(vtx)
            move = board.vertex_to_text(vtx)
            gtp_print(move)
        elif main == "time_settings":
            main_time = int(inputs[1])
            byo_time = int(inputs[2])
            byo_stones = int(inputs[3])
            time_control.time_settings(main_time, byo_time, byo_stones)
            gtp_print("")
        elif main == "time_left":
            gtp_print("")
        elif main == "lz-genmove_analyze":
            c = board.to_move
            interval = 0
            for tag in inputs[1:]:
                if tag.isdigit():
                   interval = int(tag)
                else:
                    color = tag
                    if color.lower()[:1] == "b":
                        c = BLACK
                    elif color.lower()[:1] == "w":
                        c = WHITE
            stdout_write("=\n")
            search = Search(board, net, time_control, device)
            search.analysis_tag["interval"] = interval/100
            vtx, _ = search.think(args.playouts, args.resign_threshold, args.verbose)
            board.play(vtx)
            move = board.vertex_to_text(vtx)
            stdout_write("play {}\n\n".format(move))
        elif main == "lz-analyze":
            c = board.to_move
            interval = 0
            for tag in inputs[1:]:
                if tag.isdigit():
                   interval = int(tag)
                else:
                    color = tag
                    if color.lower()[:1] == "b":
                        c = BLACK
                    elif color.lower()[:1] == "w":
                        c = WHITE
            stdout_write("=\n")
            search = Search(board, net, time_control, device)
            search.analysis_tag["interval"] = interval/100
            search.ponder(args.playouts, args.verbose)
            stdout_write("\n")
        elif main == "raw-nn":
            planes = torch.from_numpy(board.get_features()).float().to(device)
            pred, _ = net.forward(torch.unsqueeze(planes, 0))
            print_raw_nn(pred, board)
            gtp_print("")
        elif main == "planes":
            print_planes(board)
            gtp_print("")
        elif main == "gogui-analyze_commands":
            supported_list = [
                "gfx/Policy Rating/gogui-policy_rating",
                "gfx/Policy Heatmap/gogui-policy_heatmap",
                "gfx/Policy Order/gogui-policy_order",
                "gfx/Opponent Policy Rating/gogui-opp_policy_rating",
                "gfx/Opponent Policy Heatmap/gogui-opp_policy_heatmap",
                "gfx/Opponent Policy Order/gogui-opp_policy_order",
                "gfx/Soft Policy Rating/gogui-soft_policy_rating",
                "gfx/Soft Policy Heatmap/gogui-soft_policy_heatmap",
                "gfx/Soft Policy Order/gogui-soft_policy_order",
                "gfx/Optimistic Policy Rating/gogui-optimistic_policy_rating",
                "gfx/Optimistic Policy Heatmap/gogui-optimistic_policy_heatmap",
                "gfx/Optimistic Policy Order/gogui-optimistic_policy_order",
                "gfx/Ownership Heatmap/gogui-ownership_heatmap",
                "gfx/Ownership Influence/gogui-ownership_influence",
                "gfx/Ladder Heatmap/gogui-ladder_heatmap"
            ]
            out = str()
            for i in supported_list:
                out += (i + "\n")
            out = out[:-1]
            gtp_print(out)
        elif main == "gogui-policy_rating":
            planes = torch.from_numpy(board.get_features()).float().to(device)
            pred, _ = net.forward(torch.unsqueeze(planes, 0))
            prob, _, _, _, _, _, _, _, _, _ = pred
            out = gogui_policy_rating(prob, board)
            gtp_print(out)
        elif main == "gogui-policy_heatmap":
            planes = torch.from_numpy(board.get_features()).float().to(device)
            pred, _ = net.forward(torch.unsqueeze(planes, 0))
            prob, _, _, _, _, _, _, _, _, _ = pred
            out = gogui_policy_heatmap(prob, board)
            gtp_print(out)
        elif main == "gogui-policy_order":
            planes = torch.from_numpy(board.get_features()).float().to(device)
            pred, _ = net.forward(torch.unsqueeze(planes, 0))
            prob, _, _, _, _, _, _, _, _, _ = pred
            out = gogui_policy_order(prob, board)
            gtp_print(out)
        elif main == "gogui-opp_policy_rating":
            planes = torch.from_numpy(board.get_features()).float().to(device)
            pred, _ = net.forward(torch.unsqueeze(planes, 0))
            _, prob, _, _, _, _, _, _, _, _ = pred
            out = gogui_policy_rating(prob, board)
            gtp_print(out)
        elif main == "gogui-opp_policy_heatmap":
            planes = torch.from_numpy(board.get_features()).float().to(device)
            pred, _ = net.forward(torch.unsqueeze(planes, 0))
            _, prob, _, _, _, _, _, _, _, _ = pred
            out = gogui_policy_heatmap(prob, board)
            gtp_print(out)
        elif main == "gogui-opp_policy_order":
            planes = torch.from_numpy(board.get_features()).float().to(device)
            pred, _ = net.forward(torch.unsqueeze(planes, 0))
            _, prob, _, _, _, _, _, _, _, _ = pred
            out = gogui_policy_order(prob, board)
            gtp_print(out)
        elif main == "gogui-soft_policy_rating":
            planes = torch.from_numpy(board.get_features()).float().to(device)
            pred, _ = net.forward(torch.unsqueeze(planes, 0))
            _, _, prob, _, _, _, _, _, _, _ = pred
            out = gogui_policy_rating(prob, board)
            gtp_print(out)
        elif main == "gogui-soft_policy_heatmap":
            planes = torch.from_numpy(board.get_features()).float().to(device)
            pred, _ = net.forward(torch.unsqueeze(planes, 0))
            _, _, prob, _, _, _, _, _, _, _ = pred
            out = gogui_policy_heatmap(prob, board)
            gtp_print(out)
        elif main == "gogui-soft_policy_order":
            planes = torch.from_numpy(board.get_features()).float().to(device)
            pred, _ = net.forward(torch.unsqueeze(planes, 0))
            _, _, prob, _, _, _, _, _, _, _ = pred
            out = gogui_policy_order(prob, board)
            gtp_print(out)
        elif main == "gogui-optimistic_policy_rating":
            planes = torch.from_numpy(board.get_features()).float().to(device)
            pred, _ = net.forward(torch.unsqueeze(planes, 0))
            _, _, _, _, prob, _, _, _, _, _ = pred
            out = gogui_policy_rating(prob, board)
            gtp_print(out)
        elif main == "gogui-optimistic_policy_heatmap":
            planes = torch.from_numpy(board.get_features()).float().to(device)
            pred, _ = net.forward(torch.unsqueeze(planes, 0))
            _, _, _, _, prob, _, _, _, _, _ = pred
            out = gogui_policy_heatmap(prob, board)
            gtp_print(out)
        elif main == "gogui-optimistic_policy_order":
            planes = torch.from_numpy(board.get_features()).float().to(device)
            pred, _ = net.forward(torch.unsqueeze(planes, 0))
            _, _, _, _, prob, _, _, _, _, _ = pred
            out = gogui_policy_order(prob, board)
            gtp_print(out)
        elif main == "gogui-ownership_heatmap":
            planes = torch.from_numpy(board.get_features()).float().to(device)
            pred, _ = net.forward(torch.unsqueeze(planes, 0))
            _, _, _, _, _, ownership, _, _, _, _ = pred
            out = gogui_ownership_heatmap(ownership, board)
            gtp_print(out)
        elif main == "gogui-ownership_influence":
            planes = torch.from_numpy(board.get_features()).float().to(device)
            pred, _ = net.forward(torch.unsqueeze(planes, 0))
            _, _, _, _, _, ownership, _, _, _, _ = pred
            out = gogui_ownership_influence(ownership, board)
            gtp_print(out)
        elif main == "gogui-ladder_heatmap":
            laddermap = board._get_ladder_map()
            out = gogui_ladder_heatmap(laddermap, board)
            gtp_print(out)
        else:
            gtp_print("unknown command", False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json", metavar="<string>",
                        help="The setting json file name.", type=str)
    parser.add_argument("-c", "--checkpoint", metavar="<string>",
                        help="The path of checkpoint.", type=str)
    parser.add_argument("--use-swa", help="Use the SWA weights.",
                        action="store_true", default=False)
    parser.add_argument("--use-gpu", help="Use the GPU.",
                        action="store_true", default=False)
    parser.add_argument("-p", "--playouts", metavar="<integer>",
                        help="The number of playouts.", type=int, default=400)
    parser.add_argument("-r", "--resign-threshold", metavar="<float>",
                        help="Resign when winrate is less than x.", type=float, default=0.1)
    parser.add_argument("-v", "--verbose", default=False,
                        help="Dump some search verbose.", action="store_true")
    args = parser.parse_args()
    gtp_loop(args)
