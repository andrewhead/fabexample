#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import logging
import unittest
import numpy as np
import mock

from simplex.simplex import Simplex, query_ranks, Mutation


logging.basicConfig(level=logging.INFO, format="%(message)s")


class MutationTest(unittest.TestCase):

    def setUp(self):
        self.simplex = Simplex()

    def test_reflect(self):
        worst = np.array([1, 1])
        centroid = np.array([2, 2])
        reflected = self.simplex.reflect(worst, centroid, alpha=2)
        self.assertTrue(np.all(reflected == np.array([4, 4])))

    def test_reflect_default_alpha_is_1(self):
        worst = np.array([1, 1])
        centroid = np.array([2, 2])
        reflected = self.simplex.reflect(worst, centroid)
        self.assertTrue(np.all(reflected == np.array([3, 3])))

    def test_expand(self):
        worst = np.array([1, 1])
        centroid = np.array([2, 2])
        expanded = self.simplex.expand(worst, centroid, gamma=3)
        self.assertTrue(np.all(expanded == np.array([5, 5])))

    def test_expand_default_gamma_is_2(self):
        worst = np.array([1, 1])
        centroid = np.array([2, 2])
        expanded = self.simplex.expand(worst, centroid)
        self.assertTrue(np.all(expanded == np.array([4, 4])))

    def test_contract(self):
        worst = np.array([1, 1])
        centroid = np.array([2, 2])
        contracted = self.simplex.contract(worst, centroid, rho=-.3)
        self.assertTrue(np.all(contracted == np.array([1.7, 1.7])))

    def test_contract_default_rho_is_negative_half(self):
        worst = np.array([1, 1])
        centroid = np.array([2, 2])
        contracted = self.simplex.contract(worst, centroid)
        self.assertTrue(np.all(contracted == np.array([1.5, 1.5])))

    def test_reduce(self):
        best = np.array([1, 1])
        other = np.array([2, 2])
        reduced = self.simplex.reduce(best, other)
        self.assertTrue(np.all(reduced == np.array([1.5, 1.5])))

    def test_reduce_default_sigma_is_one_half(self):
        best = np.array([1, 1])
        other = np.array([2, 2])
        reduced = self.simplex.reduce(best, other, sigma=.3)
        self.assertTrue(np.all(reduced == np.array([1.3, 1.3])))


class DecisionTest(unittest.TestCase):

    def setUp(self):
        self.simplex = Simplex()

    def test_pick_expand_if_expansion_best(self):
        mut_type = self.simplex.decide_mutation(
            np.array([3, 4, 5]), 2, 1, 6
        )
        self.assertEqual(mut_type, Mutation.EXPAND)

    def test_pick_reflect_if_reflection_better_than_expand_and_best(self):
        mut_type = self.simplex.decide_mutation(
            np.array([3, 4, 5]), 1, 2, 6
        )
        self.assertEqual(mut_type, Mutation.REFLECT)

    def test_pick_reflect_if_reflection_better_than_second_worst(self):
        mut_type = self.simplex.decide_mutation(
            np.array([1, 3, 4]), 2, 5, 6
        )
        self.assertEqual(mut_type, Mutation.REFLECT)

    def test_pick_contract_if_contraction_better_than_worst(self):
        mut_type = self.simplex.decide_mutation(
            np.array([1, 2, 4]), 5, 6, 3
        )
        self.assertEqual(mut_type, Mutation.CONTRACT)

    def test_pick_reduce_if_reflection_worse_than_all(self):
        mut_type = self.simplex.decide_mutation(
            np.array([1, 2, 3]), 4, 5, 6
        )
        self.assertEqual(mut_type, Mutation.REDUCE)


class UpdatePointsTest(unittest.TestCase):

    def setUp(self):
        self.simplex = Simplex()
        self.rank_func = lambda X: np.array([x[0] for x in X])

    def test_expand_points(self):
        vertices = self.simplex.update_points(
            np.array([[1.0, 0], [2.0, 0], [3.1, 0]]),
            self.rank_func,
        )
        self.assertTrue(np.all(
            vertices - np.array([[1, 0], [2, 0], [-0.1, 0]]) < 0.0001
        ))

    def test_reflect_points(self):
        vertices = self.simplex.update_points(
            np.array([[1.0, 0], [2.0, 0], [2.9, 0]]),
            self.rank_func,
        )
        self.assertTrue(np.all(
            vertices - np.array([[1.0, 0], [2.0, 0], [float(31)/30, 0]]) < 0.0001
        ))

    def test_contract_points(self):
        vertices = self.simplex.update_points(
            np.array([[1.0, 0], [2.0, 0], [6.0, 0]]),
            lambda X: np.array([2, 3, 4, 5, 6, 1]) if len(X) == 6 else np.array([1, 2, 3]),
        )
        self.assertTrue(np.all(
            vertices - np.array([[1, 0], [2, 0], [4.5, 0]]) < 0.0001
        ))

    def test_reduce_all(self):
        vertices = self.simplex.update_points(
            np.array([[1.0, 0], [2.0, 0], [3.0, 0]]),
            lambda X: np.array([1, 2, 3, 4, 5, 6]) if len(X) == 6 else np.array([1, 2, 3]),
        )
        self.assertTrue(np.all(
            vertices - np.array([[1, 0], [1.5, 0], [2, 0]]) < 0.0001
        ))


class SimplexUtilityTest(unittest.TestCase):

    def setUp(self):
        self.simplex = Simplex()

    def test_get_centroid(self):
        c = self.simplex.centroid(np.array([[1, 1], [3, 3], [2, 2]]))
        self.assertEqual(len(c), 2)
        self.assertTrue(np.all(c == np.array([2, 2])))

    def test_get_ranks(self):
        input_mock = mock.Mock(return_value='2 1')
        with mock.patch('__builtin__.raw_input', input_mock):
            ranks = query_ranks(np.array([[1, 1], [2, 2]]))
            self.assertTrue(np.all(ranks == np.array([2, 1])))

    def test_get_ranks_skip_extraneous_input(self):
        input_mock = mock.Mock(return_value='2 1 3 4 5')
        with mock.patch('__builtin__.raw_input', input_mock):
            ranks = query_ranks(np.array([[1, 1], [2, 2]]))
            self.assertTrue(np.all(ranks == np.array([2, 1])))
