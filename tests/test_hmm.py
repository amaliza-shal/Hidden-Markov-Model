"""Tests for Viterbi and Baum-Welch, using the GaussianHMM class from the notebook.

Run from the repository root:  python -m unittest discover tests
"""
import itertools
import json
import unittest
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

NOTEBOOK = Path(__file__).resolve().parent.parent / "HMM_Activity_Recognition.ipynb"


def load_model_class():
    """Take the GaussianHMM class straight from the notebook, so the tests check the real code."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    source = next("".join(c["source"]) for c in cells
                  if c["cell_type"] == "code" and "class GaussianHMM" in "".join(c["source"]))
    namespace = {"np": np, "logsumexp": logsumexp}
    exec(source, namespace)
    return namespace["GaussianHMM"]


GaussianHMM = load_model_class()


def toy_model():
    m = GaussianHMM(2, var_floor=1e-6)
    m.means, m.vars = np.array([[0.0], [3.0]]), np.array([[1.0], [1.0]])
    m.A, m.pi = np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([0.6, 0.4])
    return m


def brute_force(m, X):
    """Exact log p(X) and best path by trying every state path."""
    log_b, log_a, log_pi = m.log_emission(X), np.log(m.A), np.log(m.pi)
    scores = {p: log_pi[p[0]] + log_b[0, p[0]] + sum(log_a[p[t - 1], p[t]] + log_b[t, p[t]] for t in range(1, len(X)))
              for p in itertools.product(range(m.K), repeat=len(X))}
    best = max(scores, key=scores.get)
    return logsumexp(list(scores.values())), np.array(best), scores[best]


def sample(m, length, rng):
    z = [rng.choice(2, p=m.pi)]
    for _ in range(length - 1):
        z.append(rng.choice(2, p=m.A[z[-1]]))
    z = np.array(z)
    return rng.normal(m.means[z, 0], np.sqrt(m.vars[z, 0]))[:, None]


X = np.array([[0.1], [2.9], [3.2], [0.2], [1.5]])


class TestViterbi(unittest.TestCase):
    def test_matches_brute_force(self):
        m = toy_model()
        path, log_p = m.viterbi(X)
        _, best_path, best_score = brute_force(m, X)
        np.testing.assert_array_equal(path, best_path)
        self.assertAlmostEqual(log_p, best_score)

    def test_is_deterministic(self):
        m = toy_model()
        np.testing.assert_array_equal(m.viterbi(X)[0], m.viterbi(X)[0])

    def test_never_uses_impossible_transition(self):
        m = toy_model()
        m.A = np.array([[1.0, 0.0], [0.5, 0.5]])
        path, log_p = m.viterbi(X)
        self.assertTrue(np.isfinite(log_p))
        self.assertNotIn((0, 1), list(zip(path[:-1], path[1:])))


class TestForward(unittest.TestCase):
    def test_likelihood_matches_brute_force(self):
        m = toy_model()
        self.assertAlmostEqual(m.forward_backward(X)[3], brute_force(m, X)[0])


class TestBaumWelch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        sequences = [sample(toy_model(), 80, rng) for _ in range(6)]
        cls.model = GaussianHMM(2, var_floor=1e-3)
        cls.model.means, cls.model.vars = np.array([[-1.0], [1.0]]), np.array([[2.0], [2.0]])
        cls.model.A, cls.model.pi = np.full((2, 2), 0.5), np.array([0.5, 0.5])
        cls.history = cls.model.fit(sequences, eps=1e-6, max_iter=500)

    def test_log_likelihood_never_falls(self):
        self.assertEqual(self.model.ll_decreases, 0)

    def test_converges(self):
        self.assertTrue(self.model.converged)

    def test_recovers_means(self):
        np.testing.assert_allclose(np.sort(self.model.means[:, 0]), [0.0, 3.0], atol=0.3)

    def test_probabilities_stay_valid(self):
        np.testing.assert_allclose(self.model.A.sum(1), 1)
        self.assertAlmostEqual(self.model.pi.sum(), 1)


if __name__ == "__main__":
    unittest.main()
