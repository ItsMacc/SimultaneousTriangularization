import numpy as np
from data import *


LOW = 1e-4
HIGH = 1e4

def generate_data(num_pairs, size_range):
    # Dictionaries to hold pairs for each type
    pair_files = {'upper': [], 'lower': [], 'similarity': [], 'random': []}

    for size in range(2, size_range + 1):
        for _ in range(num_pairs):
            # Upper triangular
            pair_files['upper'].append(__generate_upper_triangular_pair(size))

            # Lower triangular
            pair_files['lower'].append(__generate_lower_triangular_pair(size))

            # Similarity transformed
            pair_files['similarity'].append(__generate_similarity_transformed_pair(size))

            # Random
            pair_files['random'].append(__generate_random_pair(size))

    # Save each type to its own file
    for key, pairs in pair_files.items():
        save_pairs_to_file(pairs, TYPE_TO_CODE[key])

    # Return counts for reference
    counts = {key: len(pairs) for key, pairs in pair_files.items()}
    return counts


def random_log_uniform(low, high, size):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size=size))

def __generate_upper_triangular_pair(n: int) -> tuple:
    A = np.triu(random_log_uniform(LOW, HIGH, size=(n, n)))
    B = np.triu(random_log_uniform(LOW, HIGH, size=(n, n)))
    return A, B

def __generate_lower_triangular_pair(n: int) -> tuple:
    A = np.tril(random_log_uniform(LOW, HIGH, size=(n, n)))
    B = np.tril(random_log_uniform(LOW, HIGH, size=(n, n)))
    return A, B

def __generate_similarity_transformed_pair(n: int) -> tuple:
    # Generate a base triangular pair
    if np.random.rand() < 0.5:
        A, B = __generate_upper_triangular_pair(n)
    else:
        A, B = __generate_lower_triangular_pair(n)

    # Generate a random invertible matrix S
    while True:
        S = random_log_uniform(LOW, HIGH, size=(n, n))
        if np.linalg.matrix_rank(S) == n:
            break

    # Apply similarity transform
    S_inv = np.linalg.inv(S)
    A = S_inv @ A @ S
    B = S_inv @ B @ S

    return A, B

def __generate_random_pair(n: int) -> tuple:
    A = random_log_uniform(LOW, HIGH, size=(n, n))
    B = random_log_uniform(LOW, HIGH, size=(n, n))
    return A, B


def save_pairs_to_file(pairs: list, code: int):
    path = CODE_TO_PATH[code]

    with open(path, 'w') as f:
        for A, B in pairs:
            n = A.shape[0]
            f.write(f"{n}\n")
            for row in A:
                f.write(" ".join(f"{val:.8f}" for val in row) + "\n")
            f.write("\n")
            for row in B:
                f.write(" ".join(f"{val:.8f}" for val in row) + "\n")
            f.write("\n")
