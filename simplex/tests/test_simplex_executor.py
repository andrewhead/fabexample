#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import logging
import unittest
import numpy as np

from simplex.simplex import SimplexExecutor


logging.basicConfig(level=logging.INFO, format="%(message)s")


class UpdatePointsTest(unittest.TestCase):

    def test_update_points(self):
        executor = SimplexExecutor()
        updated_vertices = executor.update_points(
            np.array([[1.0, 0], [2.0, 0], [2.9, 0]]),  # vertices
            np.array([1, 2, 3]),  # vertex ranks
            0,  # reflection rank
            4,  # expansion rank
            5,  # contraction rank
        )
        self.assertTrue(np.all(
            updated_vertices - np.array([[1.0, 0], [2.0, 0], [float(31)/30, 0]]) < 0.0001
        ))


class GetNextPointsTest(unittest.TestCase):

    def test_get_next_points(self):
        executor = SimplexExecutor()
        reflection, expansion, contraction = executor.get_next_points(
            np.array([[1.0, 0], [2.0, 0], [2.9, 0]]),  # vertices
            np.array([1, 2, 3]),
        )
        self.assertTrue(np.all(
            reflection - np.array([float(31)/30, 0]) < 0.0001
        ))
        self.assertTrue(np.all(
            expansion - np.array([float(2)/15, 0]) < 0.0001
        ))
        self.assertTrue(np.all(
            contraction - np.array([float(37)/15, 0]) < 0.0001
        ))


class IsFinished(unittest.TestCase):

    def test_is_finished(self):
        executor = SimplexExecutor()
        finished = executor.is_finished(
            np.array([[1.0, 0], [1.0, 0], [1.0, 0]]),  # vertices
            np.array([1, 1, 1]),  # vertex ranks
            2,  # reflection rank
            2,  # expansion rank
            2,  # contraction rank
        )
        self.assertTrue(finished)
