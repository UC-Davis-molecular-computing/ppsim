import unittest

from ppsim import species, reactions_to_dict, Reaction
from util import assertDeepAlmostEqual


class TestCRN(unittest.TestCase):
    # tests creating protocols using CRN notation

    def setUp(self) -> None:
        self.n = 10
        self.volume = 10

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
        # self.assertDictEqual({(b, b): 2 / 5}, transitions[(a, b)])
        # self.assertDictEqual({(c, c): 3 / 5}, transitions[(b, c)])

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
        self.assertAlmostEqual(5.0, max_rate)
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
        self.assertEqual((b, b), transitions[(a, d)])
        self.assertEqual((b, b), transitions[(d, a)])
        assertDeepAlmostEqual(self, {(c, c): 2 / 3}, transitions[(a, c)])
        assertDeepAlmostEqual(self, {(c, c): 2 / 3}, transitions[(c, a)])
        assertDeepAlmostEqual(self, {(d, a): 2 / 3}, transitions[(b, a)])
        assertDeepAlmostEqual(self, {(d, b): 2 / 3}, transitions[(b, b)])
        assertDeepAlmostEqual(self, {(d, c): 2 / 3}, transitions[(b, c)])
        assertDeepAlmostEqual(self, {(d, d): 2 / 3}, transitions[(b, d)])

    def test_reversible_rxn(self) -> None:
        a, b, c, d = species('A B C D')
        rxn: Reaction = a + b | c + d
        self.assertTrue(rxn.reversible)
        transitions, max_rate = reactions_to_dict([rxn], self.n, self.volume)
        self.assertEqual(2, len(transitions))
        self.assertIn((a, b), transitions.keys())
        self.assertIn((c, d), transitions.keys())
        self.assertEqual((c, d), transitions[(a, b)])
        self.assertEqual((a, b), transitions[(c, d)])


