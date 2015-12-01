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
    compute_H, newton_rhapson, predict_f, predict_sigma


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
        self.assertAlmostEqual(res, .113548052)

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
        self.assertAlmostEqual(res, .250644809)

    def test_c_summand_is_positive_when_m_and_n_are_both_lower_point(self):
        res = c_summand(
            f=self.default_f,
            m=0,
            n=0,
            comparison=self.default_comparison,
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, .250644809)

    def test_c_summand_is_negative_when_m_is_higher_and_n_is_lower(self):
        res = c_summand(
            f=self.default_f,
            m=0,
            n=2,
            comparison=self.default_comparison,
            sigma=self.default_sigma,
        )
        self.assertAlmostEqual(res, -.250644809)

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
        self.assertAlmostEqual(res, -.250644809 / 8.0)

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
        self.assertAlmostEqual(res, -.501289618 / 8.0)

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
        self.assertAlmostEqual(res, .501289618 / 8.0)

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
        self.assertAlmostEqual(res, -.136719008)

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
            [.136719008, 0.0, -.136719008],
            [0.0, 0.0, 0.0],
            [-.136719008, 0.0, .136719008],
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
            kernelfunc=default_kernel,
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
        [.136719008, 0.0, -.136719008],
        [0.0, 0.0, 0.0],
        [-.136719008, 0.0, .136719008],
        ]
        We add these two matrices to compute the second derivative H.
        '''
        H = compute_H(
            kernelfunc=default_kernel,
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
            [-1.749178842,  1.094278880,  0.467240162],
            [1.094278880,  -1.641731230, -0.267801200],
            [0.467240162,  -0.267801200, -1.063482692],
        ]))


class NewtonRhapsonTest(NpArrayTestCase):

    def test_one_iteration_yields_expected_result(self):
        '''
        Taking our results from the computation of H in the last test,
        H^-1:
        [
        -1.066045352 -0.661325899 -0.301834093
        -0.661325899 -1.045461463 -0.027289758
        -0.301834093 -0.027289758 -1.066045352
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
        [6.879072286]
        [1.729331837]
        [1.120927715]
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
            kernelfunc=default_kernel,
            Hfunc=compute_H,
            gfunc=compute_g,
            sigma=2.0,
            maxiter=1,
        )
        self.assertAlmostEqual(f1, a([
            [-.879072286],
            [-.729331837],
            [.879072285],
        ]))

    def test_run_optimization(self):
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
            kernelfunc=default_kernel,
            Hfunc=compute_H,
            gfunc=compute_g,
            sigma=2,
            maxiter=20,
        )
        self.assertTrue(f[3][0] > f[1][0])
        self.assertTrue(f[0][0] > f[1][0])
        self.assertTrue(f[2][0] > f[1][0])
        self.assertTrue(f[4][0] > f[0][0])
        self.assertTrue(f[2][0] > f[4][0])


class ComputeExpectationTest(NpArrayTestCase):

    def test_compute_expectation(self):
        '''
        Intermediate results expected

        Kernel matrix:
        1.0 .135335283
        .135335283 1.0

        K^-1:
        1.018657360 -0.137860282
        -0.137860282  1.018657360

        Kernel vector:
        .324652467
        .882496903

        k' * Kinv:
        0.209048353 0.854205285
        '''
        expected = predict_f(
            x=a([
                [-1.0, 1.0],
                [1.0, 1.0],
            ]),
            fmap=a([
                [1.0],
                [3.0],
            ]),
            xnew=a([0.5, 1.0]),
            kernelfunc=default_kernel
        )
        self.assertAlmostEqual(expected, 2.771664208)


class ComputeSigmaTest(NpArrayTestCase):

    def test_compute_expected_sigma(self):
        '''
        To compute a realistic value of C, we do the following.
        Suppose we have two comparison: (1 > 0), (2 > 0),
        and σ[noise] = 2.0.

        Z(2 > 0) = (3 - 1) / (2√2)
        pdf(Z) = 0.31069656037692778
        cdf(Z) = 0.76024993890652326
        [(pdf^2 / cdf^2) + (pdf * Z / cdf)] = .45599496
        The matrix of h values for this comparison reads:
        1  0 -1
        0  0  0
        -1 0  1

        Then, Z(1 > 0) = (2 - 1) / (2√2)
        pdf(Z) = 0.3747715895177024
        cdf(Z) = 0.63816319508411845
        [(pdf^2 / cdf^2) + (pdf * Z / cdf)] = .760141251
        The matrix of h values for this comparison reads:
        1 -1 0
        -1 1 0
        0  0 0

        The coefficient for each term is: 1 / σ^2 = 1/4
        Summands will either be signed value of
        .627340457 / 4 = .11399874 or
        .998058129 / 4 = .190035313

        Therefore, our C matrix for this fMAP is:
        .304034053 -.190035313 -.11399874
        -.190035313 .190035313  0
        -.11399874  0           .11399874

        The inversion (C^-1) (+ I) is:
        0.792379391 0.126534115 0.081086494
        0.126534115 0.860517280 0.012948605
        0.081086494 0.012948605 0.905964901

        K is:
        1.0 .60653066 .135335283
        .60653066 1.0 .60653066
        .135335283 .60653066 1.0

        K + C^-1 is:
        1.792379391 0.733064775 0.216421777
        0.733064775 1.860517280 0.619479265
        0.216421777 0.619479265 1.905964901

        (K + C^-1)^-1 is:
         0.665294614 -0.265738423  0.010826642
        -0.265738423  0.708853612 -0.200218027
         0.010826642 -0.200218027  0.588514403

        Kernel vector:
        .324652467
        .882496903
        .882496903

        k' * (.)^-1 is:
        -0.008969320  0.362596694  0.346185245

        (.*) * k is:
        0.622585954

        k(x + 1, x + 1) will be, of course, 1, given our squared exponential kernel.
        The predicted variance is then 1 - 0.622585954 = .377414046
        '''
        sigma = predict_sigma(
            x=a([
                [-1.0, 1.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]),
            fmap=a([
                [1.0],
                [2.0],
                [3.0],
            ]),
            Cmap=a([
                [.304034053, -.190035313, -.11399874],
                [-.190035313, .190035313, 0],
                [-.11399874, 0, .11399874],
            ]),
            xnew=a([0.5, 1.0]),
            kernelfunc=default_kernel
        )
        self.assertAlmostEqual(sigma, .377414046)
