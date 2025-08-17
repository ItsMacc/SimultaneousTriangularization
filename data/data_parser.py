import numpy as np
from data import *

def load_pairs(code: int)-> list:
    pairs = []
    path = CODE_TO_PATH[code]

    with open(path, 'r') as file:
        lines = file.read().strip().split('\n')

    i = 0
    while i < len(lines):
        n = int(lines[i])
        i += 1

        A_data = []
        for _ in range(n):
            A_data.append(list(map(float, lines[i].split())))
            i += 1
        i += 1  # blank line

        B_data = []
        for _ in range(n):
            B_data.append(list(map(float, lines[i].split())))
            i += 1
        i += 1  # blank line
        pairs.append((np.array(A_data), np.array(B_data)))

    return pairs