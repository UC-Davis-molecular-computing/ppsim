import unittest

from ppsim import species, reactions_to_dict


class TestCRN(unittest.TestCase):
    # tests creating protocols using CRN notation

    def test_unit_rates_deterministic(self) -> None:
        a, b, c = species('A B C')
        rxns = [
            a + b >> 2 * b,
            b + c >> 2 * c,
            c + a >> a + a,
        ]
        transitions, max_rate = reactions_to_dict(rxns)
        self.assertAlmostEqual(1.0, max_rate)
        self.assertEqual(3, len(transitions))
        self.assertIn((a, b), transitions.keys())
        self.assertIn((b, c), transitions.keys())
        self.assertIn((c, a), transitions.keys())
        self.assertEqual((b, b), transitions[(a, b)])
        self.assertEqual((c, c), transitions[(b, c)])
        self.assertEqual((a, a), transitions[(c, a)])

    def test_nonunit_rates_deterministic(self) -> None:
        a, b, c = species('A B C')
        rxns = [
            (a + b >> 2 * b).k(2),
            (b + c >> 2 * c).k(3),
            (c + a >> a + a).k(5),
        ]
        transitions, max_rate = reactions_to_dict(rxns)
        self.assertAlmostEqual(5.0, max_rate)
        self.assertEqual(3, len(transitions))
        self.assertIn((a, b), transitions.keys())
        self.assertIn((b, c), transitions.keys())
        self.assertIn((c, a), transitions.keys())
        self.assertDictEqual({(b, b): 2 / 5}, transitions[(a, b)])
        self.assertDictEqual({(c, c): 3 / 5}, transitions[(b, c)])
        self.assertEqual((a, a), transitions[(c, a)])

    def test_unit_rates_randomized(self) -> None:
        a, b, c = species('A B C')
        rxns = [
            a + b >> 2 * b,
            a + b >> 2 * c,
            a + c >> a + a,
        ]
        transitions, max_rate = reactions_to_dict(rxns)
        self.assertAlmostEqual(2.0, max_rate)
        self.assertEqual(2, len(transitions))
        self.assertIn((a, b), transitions.keys())
        self.assertIn((a, c), transitions.keys())
        self.assertDictEqual({(b, b): 1 / 2, (c, c): 1 / 2}, transitions[(a, b)])
        self.assertDictEqual({(a, a): 1 / 2}, transitions[(a, c)])

    def test_nonunit_rates_randomized(self) -> None:
        a, b, c = species('A B C')
        rxns = [
            (a + b >> 2 * b).k(2),
            (a + b >> 2 * c).k(3),
            (a + c >> a + a).k(4),
        ]
        transitions, max_rate = reactions_to_dict(rxns)
        self.assertAlmostEqual(5.0, max_rate)
        self.assertEqual(2, len(transitions))
        self.assertIn((a, b), transitions.keys())
        self.assertIn((a, c), transitions.keys())
        self.assertDictEqual({(b, b): 2 / 5, (c, c): 3 / 5}, transitions[(a, b)])
        self.assertDictEqual({(a, a): 4 / 5}, transitions[(a, c)])

    def test_unimolecular(self) -> None:
        a, b, c, d = species('A B C D')
        rxns = [
            (a + d >> 2 * b).k(3),
            (a + c >> 2 * c).k(2),
            (b >> d).k(2),
        ]
        transitions, max_rate = reactions_to_dict(rxns)
        self.assertAlmostEqual(3.0, max_rate)
        self.assertEqual(6, len(transitions))
        self.assertIn((a, d), transitions.keys())
        self.assertIn((a, c), transitions.keys())
        self.assertIn((b, a), transitions.keys())
        self.assertIn((b, b), transitions.keys())
        self.assertIn((b, c), transitions.keys())
        self.assertIn((b, d), transitions.keys())
        self.assertEqual((b, b), transitions[(a, d)])
        self.assertDictEqual({(c, c): 2 / 3}, transitions[(a, c)])
        self.assertDictEqual({(d, a): 2 / 3}, transitions[(b, a)])
        self.assertDictEqual({(d, b): 2 / 3}, transitions[(b, b)])
        self.assertDictEqual({(d, c): 2 / 3}, transitions[(b, c)])
        self.assertDictEqual({(d, d): 2 / 3}, transitions[(b, d)])
