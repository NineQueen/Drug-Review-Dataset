"""The only hash-split seed.

- `seed(0)`: the full table. Notebooks call `split(..., SEED)`.
- `seed(1)`: K-fold on that training pool. `KFold` defaults to this.
  Stacking and parameter tuning pass it explicitly.
- `seed(2)`: MLP's 9:1 early-stop slice inside a fold fit.

Torch, NumPy, and a model's `random_state` may use `SEED`. They do not
hash-split, and they are not a substitute for the next level.
"""

SEED = 42


def seed(level: int = 0) -> int:
    """Return `SEED + level`. See the module note for which level is free."""
    if type(level) is not int or level < 0:
        raise ValueError("level must be a non-negative integer")
    return SEED + level
