"""Simple optimizers for a single neuron or small NumPy models.

This module implements two SGD variants commonly used in deep learning:

- Momentum (Polyak)
- Nesterov Accelerated Gradient (NAG)

Both are implemented in a minimal, framework-agnostic way that works with
Python floats or NumPy arrays. Each optimizer maintains its own state (the
velocity buffer) and exposes a step method to update parameters given a
current gradient.

Usage
-----
>>> import numpy as np
>>> from optimizers import Momentum, NAG
>>> w = np.array([0.0])
>>> opt = Momentum(lr=0.1, momentum=0.9)
>>> g = np.array([1.0])
>>> w = opt.step(w, g)  # returns updated weights

Design notes
------------
- stateless parameters: functions return updated parameter arrays instead of
  mutating in-place to keep it clear and testable.
- type flexibility: inputs can be float or ndarray. Internally we promote to
  numpy arrays for uniform math and maintain a velocity buffer of the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def _to_array(x: ArrayLike) -> np.ndarray:
    """Convert a float or numpy array to a numpy array (copy-free when possible)."""
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=float)


@dataclass
class Momentum:
    """SGD with classical momentum (Polyak momentum).

    Update rule (vector form):
        v_t = mu * v_{t-1} + g_t
        w_{t+1} = w_t - lr * v_t

    Where g_t is the gradient of the loss w.r.t. w at step t, mu in [0,1).

    Parameters
    ----------
    lr : float
        Learning rate (step size).
    momentum : float
        Momentum coefficient mu in [0, 1).
    nesterov : bool
        If True, performs the Nesterov look-ahead update using the classical
        decomposition: d_p = g_t + mu * v_t (where v_t is the updated velocity).
        This flag is provided for convenience; using NAG class is preferred for clarity.
    """

    lr: float = 1e-2
    momentum: float = 0.9
    nesterov: bool = False
    _v: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset internal velocity buffer."""
        self._v = None

    def step(self, w: ArrayLike, g: ArrayLike) -> np.ndarray:
        """Compute the next parameters w given current gradient g.

        Parameters
        ----------
        w : float or np.ndarray
            Current parameters.
        g : float or np.ndarray
            Current gradient of the loss w.r.t. w.

        Returns
        -------
        np.ndarray
            Updated parameters.
        """
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
    """Nesterov Accelerated Gradient (NAG).

    A specific variant of momentum SGD that applies a look-ahead gradient:

        v_t = mu * v_{t-1} + g_t
        w_{t+1} = w_t - lr * (g_t + mu * v_t)

    This matches popular deep learning frameworks' implementation of
    Nesterov momentum.
    """

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
