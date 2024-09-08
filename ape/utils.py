import json
from gzip import GzipFile
import numpy as np
from numpy.core.numeric import normalize_axis_tuple


def normalize_axis_list(axis, ndim):
    return list(normalize_axis_tuple(axis, ndim))


def load_numpy(path, *, allow_pickle: bool = True, fix_imports: bool = True, decompress: bool = False):
    if decompress:
        with GzipFile(path, 'rb') as file:
            return load_numpy(file, allow_pickle=allow_pickle, fix_imports=fix_imports)

    return np.load(path, allow_pickle=allow_pickle, fix_imports=fix_imports)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
