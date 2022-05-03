import numpy as np
import random as pr


def random_argmax(vector):
    # np.amax : 최댓값 반환
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    # pr.choice : random choice
    return pr.choice(indices)


