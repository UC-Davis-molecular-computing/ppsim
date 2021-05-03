import unittest
import numpy
from typing import Any, Dict, List
from unittest import TestCase

# taken from https://stackoverflow.com/questions/23549419/assert-that-two-dictionaries-are-almost-equal
def assertDeepAlmostEqual(test_case: TestCase, expected: Any, actual: Any,
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

# My part, using the function

# class TestMyClass(unittest.TestCase):
#     def test_dicts(self):
#         assertDeepAlmostEqual(self, {'a' : 12.4}, {'a' : 5.6 + 6.8})
#     def test_dicts_2(self):
#         dict_1 = {'a' : {'b' : [12.4, 0.3]}}
#         dict_2 = {'a' : {'b' : [5.6 + 6.8, 0.1 + 0.2]}}
#
#         assertDeepAlmostEqual(self, dict_1, dict_2)