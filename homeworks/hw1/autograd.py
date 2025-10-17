from __future__ import annotations
from typing import Callable, Set, Tuple


class Node:
    def __init__(self, data: float, _children: Tuple['Node', ...] = (), _op: str = ''):
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Node] = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Node(data={self.data}, grad={self.grad})"

    # helper to ensure Node
    @staticmethod
    def _as_node(x):
        return x if isinstance(x, Node) else Node(x)

    def __add__(self, other):
        other = Node._as_node(other)

        out = Node(self.data + other.data, (self, other), '+')

        def _backward():
            # d(out)/d(self) = 1
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = Node._as_node(other)
        out = Node(self.data * other.data, (self, other), '*')

        def _backward():
            # d(self*other)/dself = other
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def relu(self):
        out = Node(self.data if self.data > 0 else 0.0, (self,), 'relu')

        def _backward():
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)

        out._backward = _backward
        return out

    def backward(self):
        # topological order
        topo = []
        visited = set()

        def build_topo(v: Node):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # initialize grad of the output node
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
