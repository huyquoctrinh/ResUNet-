from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=0, best_metric=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra is not None:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu", strict: bool = True):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return state.get("epoch", 0), state.get("best_metric")
