import os

ROOT = os.path.dirname(os.path.abspath(__file__))
EXT = '.txt'
CODES = {
    'upper_triangular_pairs': 1,
    'lower_triangular_pairs': -1,
    'similarity_transformed_pairs': 2,
    'commuting_pairs': 3,
    'random_pairs': 69,
    'non-simultaneous-pairs': 0
}
CODE_TO_PATH = {
    1: os.path.join(ROOT, 'upper_triangular_pairs' + EXT),
    -1: os.path.join(ROOT, 'lower_triangular_pairs' + EXT),
    2: os.path.join(ROOT, 'similarity_transformed_pairs' + EXT),
    3: os.path.join(ROOT, 'commuting_pairs' + EXT),
    69: os.path.join(ROOT, 'random_pairs' + EXT),
    0: os.path.join(ROOT, 'non-simultaneous-pairs' + EXT)
}
TYPE_TO_CODE = {
    'upper': CODES['upper_triangular_pairs'],
    'lower': CODES['lower_triangular_pairs'],
    'similarity': CODES['similarity_transformed_pairs'],
    'random': CODES['random_pairs'],
    'non-simt': CODES['non-simultaneous-pairs']
}