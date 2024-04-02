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
            i = self.vertex_to_feature_index(v)
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
        # plane     38 : wave
        # plane     39 : rule
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
                features[37, self.vertex_to_feature_index(v)] = wave
                features[38, self.vertex_to_feature_index(v)] = scoring
                features[39, self.vertex_to_feature_index(v)] = side_komi/20.
                features[40, self.vertex_to_feature_index(v)] = -side_komi/20.
                features[41, self.vertex_to_feature_index(v)] = self.num_intersections/361.
                features[42, self.vertex_to_feature_index(v)] = 1.

        return np.reshape(features, (INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE))

    def superko(self):
        # Return true if the current position is superko.

        curr_hash = hash(self.state.tostring())
        s = len(self.history)
        for p in range(s-1):
            h = self.history[p]
            if hash(h.tostring()) == curr_hash:
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

def stderr_write(val):
    sys.stderr.write(val)
    sys.stderr.flush()

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

def get_vertex_from_pred(prob, board):
    prob = prob[0].cpu().detach().numpy()
    prob = get_valid_spat(prob, board)

    for idx in range(board.num_intersections):
        if not board.legal(board.index_to_vertex(idx)):
            prob[idx] = CRAZY_NEGATIVE_VALUE

    idx = np.argmax(prob).item()
    board_size = board.board_size

    if idx == board_size * board_size:
        return "pass", PASS
    x = idx % board_size
    y = idx // board_size
    return get_move_text(x, y), board.index_to_vertex(idx)

def gtp_loop(args):
    def gtp_print(res, success=True):
        if success:
            sys.stdout.write("= {}\n\n".format(res))
        else:
            sys.stdout.write("? {}\n\n".format(res))
        sys.stdout.flush()

    board = Board(BOARD_SIZE, KOMI, SCORING_AREA)
    net = load_checkpoint(args.json, args.checkpoint, args.use_swa)

    use_gpu = args.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda") if use_gpu else torch.device("cpu")
    net = net.to(device)

    if use_gpu:
        stderr_write("Enable the GPU device...\n")

    while True:
        inputs = sys.stdin.readline().strip().split()
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
                "gogui-analyze_commands"
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
            # TODO: Support MCTS?
            color = inputs[1]
            c = INVLD
            if color.lower()[:1] == "b":
                c = BLACK
            elif color.lower()[:1] == "w":
                c = WHITE
            board.to_move = c

            planes = torch.from_numpy(board.get_features()).float().to(device)
            pred, _ = net.forward(torch.unsqueeze(planes, 0))
            prob, _, _, _, _, _, _, _, _, _ = pred
            move, vtx = get_vertex_from_pred(prob, board)
            board.play(vtx)
            gtp_print(move)
        elif main == "planes":
            out = str()
            planes = board.get_features()
            for i in range(INPUT_CHANNELS - 6):
                plane = planes[i]
                out += "planes: {}\n".format(i+1)
                out += "{}\n".format(plane.astype(int))
            gtp_print(out)
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
    args = parser.parse_args()
    gtp_loop(args)
