from __future__ import annotations

import importlib
import re

_PGN_ATTR_RE = re.compile(r'^(is_fast_pgn|decode_pgn|encode_pgn)_(\d+)(?:_|$)')
_MODULE_CACHE = {}


def _get_module(name):
    match = _PGN_ATTR_RE.match(name)
    if not match:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

    module_name = f'.pgn_{match.group(2)}'
    module = _MODULE_CACHE.get(module_name)
    if module is None:
        module = importlib.import_module(module_name, __package__)
        _MODULE_CACHE[module_name] = module
    return module


def __getattr__(name):
    module = _get_module(name)
    value = getattr(module, name)
    globals()[name] = value
    return value
