#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import logging
import unittest
import sys
import numpy as np
from numpy import array as a
from numpy.testing import assert_equal, assert_almost_equal

from simplex.bayesopt import h, get_distinct_x, default_kernel,\
    get_comparison_indices, kernel_vector, kernel_matrix, c_pdf_cdf_term,\
    c_summand, compute_z, c_m_n, compute_C, b_summand, b_j, compute_b, compute_g,\
    compute_H, newton_rhapson


logging.basicConfig(level=logging.INFO, format="%(message)s")


class NpArrayTestCase(unittest.TestCase):

    def assertEqual(self, a, b):
        assert_equal(a, b)

    def assertAlmostEqual(self, a, b):
        assert_almost_equal(a, b)


class ComputeSmallHTest(NpArrayTestCase):

    # Cases to handle include:
    # 1. Derivative of (r, c) where x == r != c → 1
    # 2. Derivative of (r, c) where x == c != r → -1
    # 3. Derivative of (r, c) where x == r == c → 0
    # 4. Derivative of (r, c) where x != r, x != c → 0

    def test_compute_h_when_point_is_r(self):
        point_index = 1
        comp = a([1, 4])
        res = h(comp, point_index)
        self.assertAlmostEqual(res, 1.0)

    def test_compute_h_when_point_is_c(self):
        point_index = 4
        comp = a([1, 4])
        res = h(comp, point_index)
        self.assertAlmostEqual(res, -1.0)

    def test_compute_h_when_point_is_both_r_and_c(self):
        point_index = 2
        comp = a([2, 2])
        res = h(comp, point_index)
        self.assertAlmostEqual(res, 0.0)

    def test_compute_h_when_point_is_neither_r_nor_c(self):
        point_index = 3
        comp = a([1, 4])
        res = h(comp, point_index)
        self.assertAlmostEqual(res, 0.0)


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
            [-1.0, 0.0],
        ])
        K = kernel_matrix(default_kernel, x)
        self.assertAlmostEqual(K, a([
            [1.0, .60653066, .367879441],
            [.60653066, 1.0, .082084999],
            [.367879441, .082084999, 1.0],
        ]))


class ComputeZTest(NpArrayTestCase):

    def test_compute_z(self):
        z = compute_z(
            fr=6.0,
            fc=2.0,
            sigma=2.0
        )
        self.assertAlmostEqual(z, 1.414213562)


class ComputeCMatrixTest(NpArrayTestCase):

    def setUp(self):
        self.default_f = a([
            [6.0],
            [1.0],
            [2.0],
        ])
        self.default_comparison = a([0, 2], dtype=np.int)
        self.default_sigma = 2.0

    def test_compute_cdf_pdf_term(self):
        res = c_pdf_cdf_term(z=2)
        self.assertAlmostEqual(res, .06249979)

    def test_c_summand_is_zero_when_n_is_not_in_the_comparison(self):
        res = c_summand(
            f=self.default_f,
            m=0,
            n=1,
            comparison=self.default_comparison,
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, 0.0)

    def test_c_summand_is_zero_when_m_is_not_in_the_comparison(self):
        res = c_summand(
            f=self.default_f,
            m=1,
            n=0,
            comparison=self.default_comparison,
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, 0.0)

    def test_c_summand_is_zero_when_neither_m_nor_n_is_in_the_comparison(self):
        res = c_summand(
            f=self.default_f,
            m=1,
            n=1,
            comparison=self.default_comparison,
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, 0.0)

    def test_c_summand_is_positive_when_m_and_n_are_both_higher_point(self):
        res = c_summand(
            f=self.default_f,
            m=0,
            n=0,
            comparison=self.default_comparison,
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, .205949837)

    def test_c_summand_is_positive_when_m_and_n_are_both_lower_point(self):
        res = c_summand(
            f=self.default_f,
            m=0,
            n=0,
            comparison=self.default_comparison,
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, .205949837)

    def test_c_summand_is_negative_when_m_is_higher_and_n_is_lower(self):
        res = c_summand(
            f=self.default_f,
            m=0,
            n=2,
            comparison=self.default_comparison,
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, -.205949837)

    def test_c_entry_is_summand_over_doubled_squared_sigma_when_only_one_relevant_comparison(self):
        res = c_m_n(
            f=self.default_f,
            m=0,
            n=2,
            comparisons=a([
                [0, 2],
            ]),
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, -.205949837 / 8.0)

    def test_c_entry_doubles_with_two_positive_comparisons_where_m_and_n_are_different(self):
        res = c_m_n(
            f=self.default_f,
            m=0,
            n=2,
            comparisons=a([
                [0, 2],
                [0, 2],
            ]),
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, -.411899674 / 8.0)

    def test_c_entry_doubles_with_two_positive_comparisons_where_m_and_n_are_same(self):
        res = c_m_n(
            f=self.default_f,
            m=0,
            n=0,
            comparisons=a([
                [0, 2],
                [0, 2],
            ]),
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, .411899674 / 8.0)

    def test_c_entry_zero_if_no_relevant_comparisons(self):
        res = c_m_n(
            f=self.default_f,
            m=0,
            n=2,
            comparisons=a([
                [1, 1],
                [1, 1],
                [1, 1],
            ]),
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, 0.0)

    def test_c_entry_with_two_constrasting_comparisons(self):
        res = c_m_n(
            f=self.default_f,
            m=0,
            n=2,
            comparisons=a([
                [0, 2],
                [2, 0],
            ]),
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, -23.544537719 / 8.0)

    def test_compose_c_matrix(self):
        C = compute_C(
            f=self.default_f,
            comparisons=a([
                [0, 2],
                [2, 0],
            ]),
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(C, a([
            [2.943067215, 0.0, -2.943067215],
            [0.0, 0.0, 0.0],
            [-2.943067215, 0.0, 2.943067215],
        ]))


class ComputeGradientTest(NpArrayTestCase):

    def setUp(self):
        self.default_f = a([
            [6.0],
            [1.0],
            [2.0],
        ])
        self.default_comparison = a([0, 2], dtype=np.int)
        self.default_sigma = 2.0

    def test_b_summand_is_0_when_j_not_in_comparison(self):
        res = b_summand(
            f=self.default_f,
            j=1,
            comparison=self.default_comparison,
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, 0.0)

    def test_b_summand_is_positive_when_j_is_higher_comparison_point(self):
        res = b_summand(
            f=self.default_f,
            j=0,
            comparison=self.default_comparison,
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, .159290823)

    def test_b_summand_is_negative_when_j_is_lower_comparison_point(self):
        res = b_summand(
            f=self.default_f,
            j=2,
            comparison=self.default_comparison,
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, -.159290823)

    def test_b_j_composed_of_one_summand_if_only_one_relevant_pair(self):
        res = b_j(
            f=self.default_f,
            j=2,
            comparisons=a([
                [0, 2],
            ]),
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, -.056317811)

    def test_b_j_made_up_of_multiple_terms_if_j_in_multiple_comparisons(self):
        res = b_j(
            f=self.default_f,
            j=2,
            comparisons=a([
                [0, 2],
                [2, 1],
            ]),
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, .151312099)

    def test_b_j_is_zero_when_no_comparisons_contain_j(self):
        res = b_j(
            f=self.default_f,
            j=1,
            comparisons=a([
                [0, 2],
                [2, 0],
            ]),
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, 0.0)

    def test_compute_b_with_j_for_each_point(self):
        b = compute_b(
            f=self.default_f,
            comparisons=a([
                [0, 2],
                [2, 1],
            ]),
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(b, a([
            [.056317811],
            [-.207629909],
            [.151312099],
        ]))

    def test_compute_gradient(self):
        '''
        About this computation:
        Recall from the above test case that the kernel will be:
        [
        [1.0, .60653066, .367879441],
        [.60653066, 1.0, .082084999],
        [.367879441, .082084999, 1.0],
        ]
        It follows that the inverted kernel is:
        [
        [ 1.88589785, -1.09427888, -0.60395917],
        [-1.09427888,  1.64173123,  0.2678012 ],
        [-0.60395917,  0.2678012 ,  1.2002017 ]
        ]
        From the above example for computing b, we know that b is:
        [
        [.056317811],
        [-.207629909],
        [.151312099],
        ]
        Therefore, the expected calculation of -K^-1 * f + b is:
         -9.013189880      .056317811    -8.956872069
         4.388339650  +  -.207629909  =   4.180709741
         0.955550420      .151312099      1.106862519
        '''
        g = compute_g(
            kernel=default_kernel,
            x=a([
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, 0.0],
            ]),
            f=self.default_f,
            comparisons=a([
                [0, 2],
                [2, 1],
            ]),
            sigma=self.default_sigma
        )
        self.assertAlmostEqual(g, a([
            [-8.956872069],
            [4.180709741],
            [1.106862519],
        ]))


class ComputeHTest(NpArrayTestCase):

    def test_compute_H(self):
        '''
        This test reuse intermediate test results that can be seen in tests
        for computing the kernel, computing the gradient, and computing C.
        From these previous test cases, we know that:
        K^-1 is:
        [
        [ 1.88589785, -1.09427888, -0.60395917],
        [-1.09427888,  1.64173123,  0.2678012 ],
        [-0.60395917,  0.2678012 ,  1.2002017 ]
        ]
        and C is:
        [
        [2.943067215, 0.0, -2.943067215],
        [0.0, 0.0, 0.0],
        [-2.943067215, 0.0, 2.943067215],
        ]
        We add these two matrices to compute the second derivative H.
        '''
        H = compute_H(
            kernel=default_kernel,
            x=a([
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, 0.0],
            ]),
            f=a([
                [6.0],
                [1.0],
                [2.0],
            ]),
            comparisons=a([
                [0, 2],
                [2, 0],
            ]),
            sigma=2.0,
        )
        self.assertAlmostEqual(H, a([
            [4.828965065, -1.09427888, -3.547026385],
            [-1.09427888,  1.64173123,  0.2678012],
            [-3.547026385,  0.2678012, 4.143268915]
        ]))


class NewtonRhapsonTest(NpArrayTestCase):

    def test_one_iteration_yields_expected_result(self):
        '''
        Taking our results from the computation of H in the last test,
        H^-1:
        [
        [0.750891056 0.399854727 0.616988389]
        [0.399854727 0.828529076 0.288760930]
        [0.616988389 0.288760930 0.750891056]
        ]
        b (computed fresh):
        [
        [-.603424068]
        [0.0]
        [.603424068]
        ]
        g:
        [
        [-9.616613948]
        [4.388339650]
        [1.558974488]
        ]
        Then, we compute that H^-1 * g:
        [
        [-4.504461892]
        [0.240789374]
        [-3.495538109]
        ]
        '''
        f1 = newton_rhapson(
            x=a([
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, 0.0],
            ]),
            f0=a([
                [6.0],
                [1.0],
                [2.0],
            ]),
            comparisons=a([
                [0, 2],
                [2, 0],
            ]),
            kernel=default_kernel,
            Hfunc=compute_H,
            gfunc=compute_g,
            sigma=2.0,
            maxiter=1,
        )
        self.assertAlmostEqual(f1, a([
            [10.504461892],
            [.759210626],
            [5.495538109],
        ]))

    def test_run_for_a_while(self):
        f = newton_rhapson(
            x=a([
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, 0.0],
                [2.0, 2.0],
                [1.5, -1.0]
            ]),
            f0=a([
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
            ]),
            comparisons=a([
                [3, 1],
                [0, 1],
                [2, 1],
                [4, 0],
                [2, 4],
            ]),
            kernel=default_kernel,
            Hfunc=compute_H,
            gfunc=compute_g,
            sigma=.1,
        )
