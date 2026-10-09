"""Portable storage of experiment results: one JSON object per run, gzip-compressed (JSON Lines).

Every experiment script writes a list of flat records (dicts); NumPy arrays are stored as lists and turned back
into arrays when the file is read.  Non-finite floats are stored as the strings "inf", "-inf" and "nan" so that the
files are valid JSON and can be read from any language.
"""
import gzip
import json
import math
import numpy as np

_SPECIAL = {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}


def _enc(v):
    if isinstance(v, np.ndarray):
        return [_enc(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_enc(x) for x in v]
    if isinstance(v, (np.floating, float)):
        v = float(v)
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _dec(v):
    if isinstance(v, str) and v in _SPECIAL:
        return _SPECIAL[v]
    if isinstance(v, list):
        if v and all(isinstance(x, (int, float, str)) and (not isinstance(x, str) or x in _SPECIAL) for x in v):
            return np.array([_dec(x) for x in v])
        return [_dec(x) for x in v]
    return v


def save_records(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps({k: _enc(v) for k, v in r.items()}, separators=(",", ":")) + "\n")


def load_records(path):
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append({k: _dec(v) for k, v in json.loads(line).items()})
    return out
