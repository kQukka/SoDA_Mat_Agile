from ...logistic_env.common import *
import numpy as np


def one_hot(x, size_):
    return np.identity(size_)[x:x + 1].astype(np.float32)


def print_env_map(env_map):
    for map_ in env_map:
        print(map_)


def print_str_direct(idx_direct, num_col):
    char_direct = []
    cnt = 0
    while cnt < len(idx_direct)-1:
        txt = ''
        buf = []
        for idx in range(num_col):
            char_ = None
            if idx_direct[cnt] in IDX_ACTION:
                char_ = INITIAL_ACTION[idx_direct[cnt]]
            else:
                char_ = ' '
            txt += char_
            buf.append(char_)
            if idx < num_col-1:
                txt += ' | '
            cnt += 1
        print(txt)
        char_direct.append(buf)
    return char_direct


def get_move_p(p_state, move, max_row, max_col):
    row = p_state[0]
    col = p_state[1]

    if move == IDX_ACTION_UP:
        row -= 1
    elif move == IDX_ACTION_DOWN:
        row += 1
    elif move == IDX_ACTION_LEFT:
        col -= 1
    elif move == IDX_ACTION_RIGHT:
        col += 1
    else:
        return False

    row = 0 if row < 0 else max_row-1 if row >= max_row else row
    col = 0 if col < 0 else max_col-1 if col >= max_col else col
    return [row, col]


def get_path(map_idx_direct, p_start, p_end, max_row, max_col):
    row = p_start[0]
    col = p_start[1]

    path_ = [[row, col]]
    for _ in range(len(map_idx_direct)):
        idx_direct = map_idx_direct[(row*max_col)+col]
        p_next = get_move_p([row, col], idx_direct, max_row, max_col)
        if (p_next is False) or (p_next == [row, col]):
            break
        row = p_next[0]
        col = p_next[1]
        path_.append(p_next)
        if p_next == p_end:
            break
    return path_


def get_idx_direct(q_map, num_col):
    cnt = 0
    map_idx_direct = []
    while cnt < len(q_map)-1:
        for _ in range(num_col):
            val_max = max(q_map[cnt])
            if val_max == 0:
                map_idx_direct.append(None)
            else:
                idx = list(q_map[cnt]).index(val_max)
                if idx in IDX_ACTION:
                    map_idx_direct.append(idx)
            cnt += 1
    return map_idx_direct


def get_set_env(input_, p_start, p_end, p_wall, p_item):
    p_input = [p_start] + [p_item[ch] for ch in input_] + [p_end]

    p_start_end = []
    for idx in range(len(p_input)-1):
        p_start_end.append([p_input[idx], p_input[idx+1]])

    buf_wall = []
    for p_s, p_e in p_start_end:
        buf_p = []
        if p_s not in [p_start, p_end]:
            buf_p.append(p_s)
        if p_e not in [p_start, p_end]:
            buf_p.append(p_e)

        for p_ in p_item.values():
            if (p_ == p_s) or (p_ == p_e):
                continue
            buf_p.append(p_)
        buf_p.extend(p_wall)
        buf_wall.append(buf_p)
    return zip(p_start_end, buf_wall)


class EarlyStopping:
    def __init__(self, patience=10, ratio=100):
        self.stack = []
        self.patience = patience
        self.ratio = ratio

    def clear(self):
        del self.stack
        self.stack = []

    def check_stopping(self, value):
        self.__add_stack(value)
        num = self.__check_ratio()
        if ((num // self.patience)*100) >= self.ratio:
            return True
        return False

    def __add_stack(self, value):
        if len(self.stack) > self.patience:
            del self.stack[0]
        self.stack.append(value)

    def __check_ratio(self):
        stand = self.stack[-1]
        num = 0
        for val in self.stack:
            if stand == val:
                num += 1
        return num
