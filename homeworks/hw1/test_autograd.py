import unittest
from autograd import Node


def numeric_grad(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


class TestAutogradAll(unittest.TestCase):

    def test_add_mul_relu_backward(self):
        # Example: a=2, b=-3, c=10, d = a + b*c, e = d.relu(); e.backward()
        a = Node(2.0)
        b = Node(-3.0)
        c = Node(10.0)
        d = a + b * c
        e = d.relu()
        e.backward()

        # forward checks
        self.assertAlmostEqual(a.data, 2.0)
        self.assertAlmostEqual(b.data, -3.0)
        self.assertAlmostEqual(c.data, 10.0)
        self.assertAlmostEqual(d.data, 2.0 + (-3.0) * 10.0)

        # relu(d) == 0 for negative d; relu output grad is set to 1.0, but derivative through relu is 0
        self.assertEqual(e.data, 0.0)
        self.assertAlmostEqual(e.grad, 1.0)
        self.assertAlmostEqual(a.grad, 0.0)
        self.assertAlmostEqual(b.grad, 0.0)
        self.assertAlmostEqual(c.grad, 0.0)

    def test_simple_chain(self):
        # z = (a*b) + (a + b)
        a = Node(3.0)
        b = Node(4.0)
        z = a * b + a + b
        z.backward()

        # dz/da = b + 1 = 4 + 1 = 5
        # dz/db = a + 1 = 3 + 1 = 4
        self.assertAlmostEqual(a.grad, 5.0)
        self.assertAlmostEqual(b.grad, 4.0)

    def test_relu_positive(self):
        # ReLU should pass gradient through when input > 0
        a = Node(5.0)
        b = a.relu()
        b.backward()
        self.assertAlmostEqual(a.grad, 1.0)

    def test_radd_rmul(self):
        # Ensure scalar on left works via __radd__/__rmul__
        a = Node(2.0)
        b = 3 + a  # calls __radd__ -> should be Node(5.0)
        c = 4 * a  # calls __rmul__ -> should be Node(8.0)
        self.assertAlmostEqual(b.data, 5.0)
        self.assertAlmostEqual(c.data, 8.0)
        # backward check
        z = b * c  # (3+a)*(4*a)
        z.backward()
        # dz/da = (4*a)*(1) + (3+a)*(4) = 4*a + 4*(3+a) = 4*2 + 4*(5) = 8 + 20 = 28
        self.assertAlmostEqual(a.grad, 28.0)

    def test_multiple_consumers(self):
        # Node used in multiple places should accumulate gradients
        a = Node(1.5)
        b = a * a  # a^2
        c = a + a  # 2a
        z = b + c  # a^2 + 2a
        z.backward()
        # dz/da = 2a + 2
        expected = 2 * a.data + 2.0
        self.assertAlmostEqual(a.grad, expected)

    def test_numeric_check_mul_add(self):
        # Compare analytic gradient to finite differences for f(a,b) = a*b + a + b
        def f_scalar(ad):
            a = Node(ad)
            b = Node(2.3)
            z = a * b + a + b
            z.backward()
            return z.data, a.grad, b.grad

        a0 = 3.14159
        z0, ag, bg = f_scalar(a0)

        def f_a(x):
            return x * 2.3 + x + 2.3

        numeric_da = numeric_grad(f_a, a0)
        self.assertAlmostEqual(ag, numeric_da, places=4)


if __name__ == '__main__':
    unittest.main()
