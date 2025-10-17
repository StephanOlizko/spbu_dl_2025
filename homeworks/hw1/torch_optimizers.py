from __future__ import annotations
from typing import Iterable, Optional
import torch
from torch.optim import Optimizer


class _BaseMomentum(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group.get("momentum", 0.0)
            wd = group.get("weight_decay", 0.0)
            nesterov = group.get("nesterov", False)

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if wd != 0.0:
                    d_p = d_p.add(p, alpha=wd)

                state = self.state[p]
                if mu != 0.0:
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(mu).add_(d_p)

                    if nesterov:
                        # d_p = g + mu * v_t
                        d_p = d_p.add(buf, alpha=mu)
                    else:
                        # d_p = v_t
                        d_p = buf

                p.add_(d_p, alpha=-lr)

        return loss


class Momentum(_BaseMomentum):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-2, momentum: float = 0.9, weight_decay: float = 0.0) -> None:
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=False)


class NAG(_BaseMomentum):

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-2, momentum: float = 0.9, weight_decay: float = 0.0) -> None:
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)


__all__ = ["Momentum", "NAG"]
