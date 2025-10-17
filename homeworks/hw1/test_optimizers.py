import unittest
import numpy as np

from optimizers import Momentum, NAG


class TestMomentumAndNAG(unittest.TestCase):
    def test_momentum_update_scalar(self):
        w = 0.0
        g = 1.0
        opt = Momentum(lr=0.1, momentum=0.9)
        # First step: v=1.0, w= -0.1
        w = opt.step(w, g)
        self.assertAlmostEqual(w.item(), -0.1, places=7)
        # Second step with same grad: v=0.9*1 + 1 = 1.9; w = -0.1 - 0.1*1.9 = -0.29
        w = opt.step(w, g)
        self.assertAlmostEqual(w.item(), -0.29, places=7)

    def test_nag_update_scalar(self):
        w = 0.0
        g = 1.0
        opt = NAG(lr=0.1, momentum=0.9)
        # First step: v=1.0; d = g + mu*v = 1 + 0.9*1 = 1.9; w = -0.19
        w = opt.step(w, g)
        self.assertAlmostEqual(w.item(), -0.19, places=7)
        # Second: v=0.9*1 + 1 = 1.9; d = 1 + 0.9*1.9 = 2.71; w = -0.19 - 0.271 = -0.461
        w = opt.step(w, g)
        self.assertAlmostEqual(w.item(), -0.461, places=7)

    def test_momentum_converges_quadratic(self):
        # Minimize f(w) = (w-3)^2; grad = 2*(w-3)
        opt = Momentum(lr=0.1, momentum=0.9)
        w = np.array([10.0])
        for _ in range(200):
            g = 2 * (w - 3.0)
            w = opt.step(w, g)
        self.assertTrue(np.allclose(w, np.array([3.0]), atol=1e-2))

    def test_nag_converges_quadratic(self):
        opt = NAG(lr=0.1, momentum=0.9)
        w = np.array([10.0])
        for _ in range(150):
            g = 2 * (w - 3.0)
            w = opt.step(w, g)
        self.assertTrue(np.allclose(w, np.array([3.0]), atol=1e-2))

    def test_momentum_nesterov_flag_equivalence(self):
        # Momentum with nesterov=True should match NAG on identical sequence
        w1 = np.array([0.5])
        w2 = np.array([0.5])
        opt1 = Momentum(lr=0.05, momentum=0.8, nesterov=True)
        opt2 = NAG(lr=0.05, momentum=0.8)
        rng = np.random.default_rng(42)
        for _ in range(50):
            g = rng.normal(size=w1.shape)
            w1 = opt1.step(w1, g)
            w2 = opt2.step(w2, g)
            self.assertTrue(np.allclose(w1, w2, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
