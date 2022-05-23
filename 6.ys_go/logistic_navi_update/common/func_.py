import os
import csv
import numpy as np
import random as pr


def random_argmax(vector):
    # np.amax : 최댓값 반환
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    # pr.choice : random choice
    return pr.choice(indices)


def save(path, str_):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        for s in str_:
            wr.writerow(s)


def edit(path, str_):
    with open(path, 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        for s in str_:
            wr.writerow(s)


def load(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        rd = csv.reader(f)
        data = []
        for s in rd:
            data.append([float(v) for v in s])
    return data


def create_dir(path):
    try:
        if os.path.exists(path):
            return False
        os.makedirs(path)
    except:
        return False
    return True


def make_path(path, name_file, extension):
    buf_path = f'{path}/{name_file}{extension}'
    if os.path.exists(path):
        for idx in range(100):
            name_file_ = f'{name_file}_{idx}'
            buf_path = f'{path}/{name_file_}{extension}'
            if not os.path.exists(buf_path):
                return buf_path
        return False
    return buf_path




