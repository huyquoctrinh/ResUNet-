import copy
import os
from pathlib import Path

import yaml


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    base_key = "_base_"
    if base_key in cfg:
        base_path = (path.parent / cfg.pop(base_key)).resolve()
        base_cfg = _load_yaml(base_path)
        cfg = _deep_update(base_cfg, cfg)
    return cfg


def _parse_override(s: str):
    if "=" not in s:
        raise ValueError(f"override must be key=value, got: {s}")
    key, val = s.split("=", 1)
    try:
        val = yaml.safe_load(val)
    except Exception:
        pass
    return key.strip(), val


def _apply_override(cfg: dict, key: str, val):
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        if p not in d or not isinstance(d[p], dict):
            d[p] = {}
        d = d[p]
    d[parts[-1]] = val


class DotDict(dict):
    def __getattr__(self, k):
        if k in self:
            v = self[k]
            if isinstance(v, dict):
                return DotDict(v)
            return v
        raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def _to_dotdict(obj):
    if isinstance(obj, dict):
        return DotDict({k: _to_dotdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_dotdict(v) for v in obj]
    return obj


def load_config(path, overrides=None):
    path = Path(path).resolve()
    cfg = _load_yaml(path)
    if overrides:
        for o in overrides:
            k, v = _parse_override(o)
            _apply_override(cfg, k, v)
    cfg["_config_path"] = str(path)
    return _to_dotdict(cfg)
