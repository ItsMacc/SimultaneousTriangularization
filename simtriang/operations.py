import numpy as np
from numpy import floating
from scipy.optimize import minimize
from typing import Any

# CONSTANTS
threshold = 2.22044605e-22
machine_epsilon = 1.1920929e-7
condition_number_threshold = 10e14


def score(entries: list) -> float | floating[Any]:
    """
    Assign a score to the triangular entries by calculating the Euclidean
    norm (L2 norm). This score represents the magnitude of the triangular
    entries.

    :param entries: A 1-D array representing the triangular entries in a column
    :return: The Euclidean norm of the entries. Returns 0 if input is invalid or empty.
    """
    if entries is None or entries.size == 0:
        return 0.0
    return np.linalg.norm(entries)


def change_column(M: np.ndarray, v: np.ndarray, col: int) -> np.ndarray:
    """
    Replaces a specific column in matrix M with the values from vector v.

    :param M: The matrix whose column we want to change
    :param v: The vector containing the new values to insert
    :param col: The column index in the matrix that will be updated by v
    :return: The matrix with the specified column replaced by vector v
    """
    # Replace the column with vector v
    M[:, col] = v
    return M


def upper_triangle_entries(T: np.ndarray, col: int) -> np.ndarray | None:
    """
    Extracts the upper triangular entries from a specified column of matrix T.

    :param T: The matrix whose upper-triangular entries will be extracted
    :param col: The column index from which the entries will be extracted
    :return: An array containing the entries below the diagonal in the column,
             or None if the column is the last one (no entries below diagonal).
    """
    if col == T.shape[0] - 1:
        return None
    return T[col + 1:, col]  # Extract only entries below diagonal


def qr_decomposition(M: np.ndarray) -> np.ndarray:
    """
    Performs a QR decomposition on a selected matrix M.

    :param M: The matrix to be decomposed
    :return: The Q matrix from the QR decomposition with small entries zeroed out
    """
    # Perform QR decomposition
    Q, _ = np.linalg.qr(M, mode='complete')

    # Zero out very small entries for numerical stability
    Q[np.abs(Q) < machine_epsilon] = 0.0

    return Q


def generate_random_vector(size: int) -> np.ndarray:
    """
    Generates a random normalized vector of a given size. Each entry is a real number.

    :param size: The size of the vector to generate
    :return: A unit vector of the specified size
    """
    x = np.random.randn(size)
    return x / np.linalg.norm(x)


def perturbed_matrix(U: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Computes the perturbed matrix by applying a similarity transformation
    and extracting only the strictly upper triangular part.

    This isolates the strictly upper triangular components in a transformed
    basis and maps them back to the original space as a perturbation term.

    :param U: The unitary matrix used for similarity transformation
    :param M: The matrix whose perturbation we want to find
    :return: The perturbation matrix
    """
    U_M_U = U.T @ M @ U
    return -(U @ np.tril(U_M_U, -1) @ U.T)


def triangularization_defect(M: np.ndarray) -> float:
    """
    Computes the "defect" of the triangularization of a matrix, defined as
    the maximum absolute value of the entries in the matrix

    :param M: The matrix whose defect is to be measured
    :return: The maximum absolute value of the entries of the matrix
    """
    return np.max(np.abs(M))


def column_cost(v: np.ndarray,
                i: int,
                M1: np.ndarray,
                M2: np.ndarray,
                U: np.ndarray) -> float:
    """
    Computes the cost for a given vector `v` in terms of the triangularization
    quality of matrices M1 and M2 when replacing the i-th column of U.

    :param v: The vector to replace the `i`-th column of the unitary matrix U
    :param i: The column index to change
    :param M1: The first matrix for triangularization
    :param M2: The second matrix for triangularization
    :param U: The current unitary matrix
    :return: The combined squared norm of strictly upper-triangular entries
             from both transformed matrices after the column change
    """
    # Replace i-th column and re-orthogonalize
    U_temp = change_column(U, v, i)
    U_temp = qr_decomposition(U_temp)

    # Compute similarity transforms
    T1 = U_temp.T @ M1 @ U_temp
    T2 = U_temp.T @ M2 @ U_temp

    # Get strictly upper-triangular entries from the i-th column
    entries1 = upper_triangle_entries(T1, i)
    entries2 = upper_triangle_entries(T2, i)

    # Return sum of squares of L2 norms as cost
    return score(entries1)**2 + score(entries2)**2


def joint_triangularization_defect(M1: np.ndarray, M2: np.ndarray) -> tuple:
    """
    Computes an approximate unitary matrix that jointly triangularizes two
    matrices M1 and M2 by minimizing the joint triangularization defect,
    defined as the maximum magnitude of strictly upper-triangular entries
    after transformation.

    The method iteratively optimizes each column of the unitary matrix to
    minimize the sum of squared magnitudes of the upper-triangular parts of
    the transformed matrices. The defect measures how close the pair
    (M1, M2) is to being simultaneously triangularizable.

    :param M1: The first square matrix for triangularization
    :param M2: The second square matrix for triangularization
    :return: A tuple containing:
        - The minimal triangularization defect (float)
        - The approximate unitary matrix U (numpy.ndarray)
        - The perturbation matrix delta_M1 (numpy.ndarray)
        - The perturbation matrix delta_M2 (numpy.ndarray)

    Limitations:
    - Supports only real matrices and orthogonal transforms; complex matrices are not supported.
    - Uses heuristic local optimization; results may depend on random initialization.
    - Computational cost increases with matrix size due to iterative column-wise optimization.
    - Assumes square matrices of the same dimension.
    - Does not guarantee exact simultaneous triangularization if none exists.
    """
    # Convert matrices to float64 for improved numerical precision and stability
    M1 = M1.astype(np.float64)
    M2 = M2.astype(np.float64)

    # Calculate norms for M1 and M2 for normalization
    norm1 = np.linalg.norm(M1, ord='fro')
    norm2 = np.linalg.norm(M2, ord='fro')
    scale = max(norm1, norm2)

    # Normalize matrices if the pair is ill-conditioned
    if max(np.linalg.cond(M1), np.linalg.cond(M2)) > condition_number_threshold:
        M1 = M1 / scale
        M2 = M2 / scale

    # Define variables
    SIZE = M1.shape[0]
    U = np.eye(SIZE)
    MIN_EPSILON = 0

    # Iteratively optimize each column of U
    for i in range(SIZE - 1):
        best_cost = float('inf')
        best_v = None

        # Multiple random initializations to avoid local minima (emperical)
        for _ in range(int(6 * SIZE**1.2)):
            v = generate_random_vector(SIZE)

            # Minimize the column cost
            res = minimize(
                column_cost,
                v,
                args=(i, M1, M2, U),
                method='L-BFGS-B',
                options={'maxiter': 200*SIZE, 'gtol': threshold, 'ftol': threshold}
            )

            cost = res.fun

            if cost < best_cost:
                best_cost = cost
                best_v = res.x

            # Stop early in case solution is optimal
            if best_cost <= threshold / SIZE:
                break

        # Update U with the best vector found for this column and re-orthogonalize
        U = change_column(U, best_v, i)
        U = qr_decomposition(U)

    delta_M1 = perturbed_matrix(U, M1)
    delta_M2 = perturbed_matrix(U, M2)

    MIN_EPSILON += max(triangularization_defect(delta_M1),
                       triangularization_defect(delta_M2))

    return MIN_EPSILON, U, delta_M1, delta_M2


def print_matrix(label: str, matrix: np.ndarray, decimals: int=8) -> None:
    """
    Prints a matrix in a human-readable format.

    :param label: Name/Label for the matrix
    :param matrix: The matrix to print
    :param decimals: The number of decimals to display (default 8)
    """
    print(f"{label}:")
    for row in matrix:
        formatted_row = ", ".join(f"{val:.{decimals}f}" for val in row)
        print(f"\t[{formatted_row}]")
