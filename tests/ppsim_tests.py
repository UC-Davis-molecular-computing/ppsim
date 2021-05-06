import unittest
from typing import List, Dict, Any

import numpy
from ppsim import species, reactions_to_dict, Reaction, Simulation


class TestBasicProtocols(unittest.TestCase):

    def test_fratricide(self) -> None:
        l, f = 'L', 'F'
        fratricide = {(l, l): (l, f)}
        sim = Simulation({l: 20}, fratricide)
        sim.run()
        self.assertEqual(sim.config_dict, {l: 1, f: 19})
        self.assertEqual('', sim.enabled_reactions)


class TestCRN(unittest.TestCase):
    # tests creating protocols using CRN notation

    def setUp(self) -> None:
        self.n = 10
        self.volume = 10
        self.bimol_correction = (self.n - 1) / (2 * self.volume)

    def test_unit_rates_deterministic(self) -> None:
        a, b, c = species('A B C')
        rxns = [
            a + b >> 2 * b,
            b + c >> 2 * c,
            c + a >> a + a,
        ]
        transitions, max_rate = reactions_to_dict(rxns, self.n, self.volume)
        self.assertAlmostEqual(0.45, max_rate)
        self.assertEqual(6, len(transitions))
        self.assertIn((a, b), transitions.keys())
        self.assertIn((b, a), transitions.keys())
        self.assertIn((b, c), transitions.keys())
        self.assertIn((c, b), transitions.keys())
        self.assertIn((c, a), transitions.keys())
        self.assertIn((a, c), transitions.keys())
        self.assertEqual((b, b), transitions[(a, b)])
        self.assertEqual((b, b), transitions[(b, a)])
        self.assertEqual((c, c), transitions[(b, c)])
        self.assertEqual((c, c), transitions[(c, b)])
        self.assertEqual((a, a), transitions[(c, a)])
        self.assertEqual((a, a), transitions[(a, c)])

    def test_nonunit_rates_deterministic(self) -> None:
        a, b, c = species('A B C')
        rxns = [
            (a + b >> 2 * b).k(2),
            (b + c >> 2 * c).k(3),
            (c + a >> a + a).k(5),
        ]
        transitions, max_rate = reactions_to_dict(rxns, self.n, self.volume)
        self.assertAlmostEqual(2.25, max_rate)
        self.assertEqual(6, len(transitions))
        self.assertIn((a, b), transitions.keys())
        self.assertIn((b, c), transitions.keys())
        self.assertIn((c, a), transitions.keys())
        self.assertIn((b, a), transitions.keys())
        self.assertIn((c, b), transitions.keys())
        self.assertIn((a, c), transitions.keys())
        assertDeepAlmostEqual(self, {(b, b): 2 / 5}, transitions[(a, b)])
        assertDeepAlmostEqual(self, {(b, b): 2 / 5}, transitions[(b, a)])
        assertDeepAlmostEqual(self, {(c, c): 3 / 5}, transitions[(b, c)])
        assertDeepAlmostEqual(self, {(c, c): 3 / 5}, transitions[(c, b)])
        self.assertEqual((a, a), transitions[(c, a)])
        self.assertEqual((a, a), transitions[(a, c)])

    def test_unit_rates_randomized(self) -> None:
        a, b, c = species('A B C')
        rxns = [
            a + b >> 2 * b,
            a + b >> 2 * c,
            a + c >> a + a,
        ]
        transitions, max_rate = reactions_to_dict(rxns, self.n, self.volume)
        self.assertAlmostEqual(0.9, max_rate)
        self.assertEqual(4, len(transitions))
        self.assertIn((a, b), transitions.keys())
        self.assertIn((b, a), transitions.keys())
        self.assertIn((a, c), transitions.keys())
        self.assertIn((c, a), transitions.keys())
        assertDeepAlmostEqual(self, {(b, b): 1 / 2, (c, c): 1 / 2}, transitions[(a, b)])
        assertDeepAlmostEqual(self, {(b, b): 1 / 2, (c, c): 1 / 2}, transitions[(b, a)])
        assertDeepAlmostEqual(self, {(a, a): 1 / 2}, transitions[(a, c)])
        assertDeepAlmostEqual(self, {(a, a): 1 / 2}, transitions[(c, a)])

    def test_nonunit_rates_randomized(self) -> None:
        a, b, c = species('A B C')
        rxns = [
            (a + b >> 2 * b).k(2),
            (a + b >> 2 * c).k(3),
            (a + c >> a + a).k(4),
        ]
        transitions, max_rate = reactions_to_dict(rxns, self.n, self.volume)
        self.assertAlmostEqual(2.25, max_rate)
        self.assertEqual(4, len(transitions))
        self.assertIn((a, b), transitions.keys())
        self.assertIn((b, a), transitions.keys())
        self.assertIn((a, c), transitions.keys())
        self.assertIn((c, a), transitions.keys())
        assertDeepAlmostEqual(self, {(b, b): 2 / 5, (c, c): 3 / 5}, transitions[(a, b)])
        assertDeepAlmostEqual(self, {(b, b): 2 / 5, (c, c): 3 / 5}, transitions[(b, a)])
        assertDeepAlmostEqual(self, {(a, a): 4 / 5}, transitions[(a, c)])
        assertDeepAlmostEqual(self, {(a, a): 4 / 5}, transitions[(c, a)])

    def test_unimolecular(self) -> None:
        a, b, c, d = species('A B C D')
        rxns = [
            (a + d >> 2 * b).k(3),
            (a + c >> 2 * c).k(2),
            (b >> d).k(2),
        ]
        transitions, max_rate = reactions_to_dict(rxns, self.n, self.volume)
        self.assertAlmostEqual(2.0, max_rate)
        self.assertEqual(8, len(transitions))
        self.assertIn((a, d), transitions.keys())
        self.assertIn((d, a), transitions.keys())
        self.assertIn((a, c), transitions.keys())
        self.assertIn((c, a), transitions.keys())
        self.assertIn((b, a), transitions.keys())
        self.assertIn((b, b), transitions.keys())
        self.assertIn((b, c), transitions.keys())
        self.assertIn((b, d), transitions.keys())
        assertDeepAlmostEqual(self, {(b, b): 3 / 2 * self.bimol_correction}, transitions[(a, d)])
        assertDeepAlmostEqual(self, {(b, b): 3 / 2 * self.bimol_correction}, transitions[(d, a)])
        assertDeepAlmostEqual(self, {(c, c): 2 / 2 * self.bimol_correction}, transitions[(a, c)])
        assertDeepAlmostEqual(self, {(c, c): 2 / 2 * self.bimol_correction}, transitions[(c, a)])
        assertDeepAlmostEqual(self, (d, a), transitions[(b, a)])
        assertDeepAlmostEqual(self, (d, b), transitions[(b, b)])
        assertDeepAlmostEqual(self, (d, c), transitions[(b, c)])
        assertDeepAlmostEqual(self, (d, d), transitions[(b, d)])

    def test_reversible_rxn(self) -> None:
        a, b, c, d = species('A B C D')
        rxn: Reaction = a + b | c + d
        self.assertTrue(rxn.reversible)
        transitions, max_rate = reactions_to_dict([rxn], self.n, self.volume)
        self.assertEqual(4, len(transitions))
        self.assertIn((a, b), transitions.keys())
        self.assertIn((c, d), transitions.keys())
        self.assertEqual((c, d), transitions[(a, b)])
        self.assertEqual((a, b), transitions[(c, d)])



# taken from https://stackoverflow.com/questions/23549419/assert-that-two-dictionaries-are-almost-equal
def assertDeepAlmostEqual(test_case: unittest.TestCase, expected: Any, actual: Any,
                          *args: List, **kwargs: Dict) -> None:
    """
    Assert that two complex structures have almost equal contents.

    Compares lists, dicts and tuples recursively. Checks numeric values
    using test_case's :py:meth:`unittest.TestCase.assertAlmostEqual` and
    checks all other values with :py:meth:`unittest.TestCase.assertEqual`.
    Accepts additional positional and keyword arguments and pass those
    intact to assertAlmostEqual() (that's how you specify comparison
    precision).

    Usage:

    .. code-block:: python

        from util import assertDeepAlmostEqual

        assertDeepAlmostEqual(self, expected, actual)

    Args:
        test_case: TestCase object on which we can call all of the basic 'assert' methods.
        expected: expected value
        actual: actual value
        *args: args to pass to TestCase.assertAlmostEqual
        **kwargs: kwargs to pass to TestCase.assertAlmostEqual
    """
    is_root = not '__trace' in kwargs
    trace = kwargs.pop('__trace', 'ROOT')
    try:
        if isinstance(expected, (int, float, numpy.long, complex)):
            test_case.assertAlmostEqual(expected, actual, *args, **kwargs)
        elif isinstance(expected, (list, tuple, numpy.ndarray)):
            test_case.assertEqual(len(expected), len(actual))
            for index in range(len(expected)):
                v1, v2 = expected[index], actual[index]
                assertDeepAlmostEqual(test_case, v1, v2,
                                      __trace=repr(index), *args, **kwargs)
        elif isinstance(expected, dict):
            test_case.assertEqual(set(expected), set(actual))
            for key in expected:
                assertDeepAlmostEqual(test_case, expected[key], actual[key],
                                      __trace=repr(key), *args, **kwargs)
        else:
            test_case.assertEqual(expected, actual)
    except AssertionError as exc:
        exc.__dict__.setdefault('traces', []).append(trace)
        if is_root:
            trace = ' -> '.join(reversed(exc.traces))
            exc = AssertionError("%s\nTRACE: %s" % (exc.message, trace))
        raise exc