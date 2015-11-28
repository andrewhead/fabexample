#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import logging
import unittest
import sys
from numpy import array as a
from numpy.testing import assert_equal, assert_almost_equal

from simplex.bayesopt import h, get_distinct_x, default_kernel,\
    get_comparison_indices, kernel_vector, kernel_matrix


logging.basicConfig(level=logging.INFO, format="%(message)s")


class NpArrayTestCase(unittest.TestCase):

    def assertEqual(self, a, b):
        assert_equal(a, b)

    def assertAlmostEqual(self, a, b):
        assert_almost_equal(a, b)


class ComputeHTest(NpArrayTestCase):

    # Cases to handle include:
    # 1. Derivative of (r, c) where x == r != c → 1
    # 2. Derivative of (r, c) where x == c != r → -1
    # 3. Derivative of (r, c) where x == r == c → 0
    # 4. Derivative of (r, c) where x != r, x != c → 0

    def test_compute_h_when_point_is_r(self):
        # Each element here is an x-position---a point in input space
        points = a([1.0, 2.0, 3.0, 4.0])
        comp = a([1.0, 0.0])
        res = h(comp, points)
        self.assertTrue(all(res == a([1.0, 0.0, 0.0, 0.0])))

    def test_compute_h_when_point_is_c(self):
        points = a([1.0, 2.0, 3.0, 4.0])
        comp = a([0.0, 2.0])
        res = h(comp, points)
        self.assertTrue(all(res == a([0.0, -1.0, 0.0, 0.0])))

    def test_compute_h_when_point_is_both_r_and_c(self):
        points = a([1.0, 2.0, 3.0, 4.0])
        comp = a([3.0, 3.0])
        res = h(comp, points)
        self.assertTrue(all(res == a([0.0, 0.0, 0.0, 0.0])))

    def test_compute_h_when_point_is_neither_r_nor_c(self):
        points = a([1.0, 2.0, 3.0, 4.0])
        comp = a([0.0, 0.0])
        res = h(comp, points)
        self.assertTrue(all(res == a([0.0, 0.0, 0.0, 0.0])))


class GetDistinctXTest(NpArrayTestCase):

    def test_get_distinct_x_from_comparisons(self):
        comp = a([
            [[0.0], [1.0]],
            [[2.0], [3.0]]
        ])
        x = get_distinct_x(comp)
        self.assertAlmostEqual(x, a([[0.0], [1.0], [2.0], [3.0]]))

    def test_skip_repetitions_within_comparison(self):
        comp = a([
            [[0.0], [1.0]],
            [[2.0], [2.0]]
        ])
        x = get_distinct_x(comp)
        self.assertAlmostEqual(x, a([[0.0], [1.0], [2.0]]))

    def test_skip_repetitions_across_comparisons(self):
        comp = a([
            [[1.0], [2.0]],
            [[2.0], [3.0]]
        ])
        x = get_distinct_x(comp)
        self.assertAlmostEqual(x, a([[1.0], [2.0], [3.0]]))

    def test_get_distinct_x_from_2_dimensional_input_data(self):
        comp = a([
            [[1.0, 2.0], [2.0, 2.0]],
            [[2.0, 2.0], [3.0, 3.0]]
        ])
        x = get_distinct_x(comp)
        self.assertAlmostEqual(x, a([[1.0, 2.0], [2.0, 2.0], [3.0, 3.0]]))


class ComparisonsToIndicesTest(NpArrayTestCase):

    def test_get_indices_for_comparisons(self):
        comp = a([
            [[1.0, 2.5], [3.0, 2.0]],
            [[1.0, 2.5], [1.0, 2.5]],
            [[2.0, 3.0], [3.0, 2.0]]
        ])
        x = a([[1.0, 2.5], [3.0, 2.0], [2.0, 3.0]])
        indices = get_comparison_indices(x, comp)
        self.assertEqual(indices, a([
            [0, 1],
            [0, 0],
            [2, 1],
        ]))


class KernelTest(NpArrayTestCase):

    def test_default_kernel_computation_yields_1_when_same(self):
        x1 = a([1.0])
        x2 = a([1.0])
        k = default_kernel(x1, x2)
        self.assertAlmostEqual(k, 1.0)

    def test_default_kernel_computation_yields_0_when_very_different(self):
        x1 = a([1.0])
        x2 = a([float(sys.maxint)])
        k = default_kernel(x1, x2)
        self.assertAlmostEqual(k, 0.0)

    def test_default_kernel_computation_with_specific_computations(self):
        x1 = a([1.0])
        x2 = a([3.0])
        k = default_kernel(x1, x2)
        self.assertAlmostEqual(k, .135335283)

    def test_default_kernel_computation_on_2d_points(self):
        x1 = a([1.0, 2.0])
        x2 = a([3.0, 4.0])
        k = default_kernel(x1, x2)
        self.assertAlmostEqual(k, .018315639)

    def test_compute_kernel_vector(self):
        x = a([
            [0.0, 1.0],
            [1.0, 1.0]
        ])
        xnew = a([0.0, 0.0])
        k = kernel_vector(default_kernel, x, xnew)
        self.assertAlmostEqual(k, a([[.60653066], [.367879441]]))

    def test_compute_kernel_matrix(self):
        x = a([
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        K = kernel_matrix(default_kernel, x)
        self.assertAlmostEqual(K, a([
            [1.0, .60653066],
            [.60653066, 1.0],
        ]))
