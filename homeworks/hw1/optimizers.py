from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def _to_array(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=float)


@dataclass
class Momentum:

    lr: float = 1e-2
    momentum: float = 0.9
    nesterov: bool = False
    _v: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._v = None

    def step(self, w: ArrayLike, g: ArrayLike) -> np.ndarray:
        w = _to_array(w)
        g = _to_array(g)
        if self._v is None:
            self._v = np.zeros_like(w)

        # v_t = mu * v_{t-1} + g_t
        self._v = self.momentum * self._v + g

        if self.nesterov:
            # Nesterov-style gradient: g_t + mu * v_t
            d_p = g + self.momentum * self._v
            return w - self.lr * d_p

        # Classical momentum
        return w - self.lr * self._v


@dataclass
class NAG:
    lr: float = 1e-2
    momentum: float = 0.9
    _v: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._v = None

    def step(self, w: ArrayLike, g: ArrayLike) -> np.ndarray:
        w = _to_array(w)
        g = _to_array(g)
        if self._v is None:
            self._v = np.zeros_like(w)

        # v_t = mu * v_{t-1} + g_t
        self._v = self.momentum * self._v + g
        # NAG look-ahead step
        d_p = g + self.momentum * self._v
        return w - self.lr * d_p


__all__ = ["Momentum", "NAG"]
