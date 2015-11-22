#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import logging
import numpy as np
from enum import Enum


logging.basicConfig(level=logging.INFO, format="%(message)s")


def query_ranks(points):
    ''' Request to a user for ranks for a set of points through stdin '''
    # Expects space-delimited ranks from input
    print "Current points:"
    for p in points:
        print '*', p
    print "Enter a rank for each (space-delimited):",
    string = raw_input()
    ranks = np.array([int(_) for _ in string.split(' ')])
    ranks_for_points = ranks[:len(points)]
    return ranks_for_points


class Simplex(object):
    ''' Nelder-Mead method for numerically determining the optimum. '''

    def centroid(self, points):
        return np.average(points, axis=0)

    def reflect(self, worst, centroid, alpha=1):
        return centroid + (centroid - worst) * alpha

    def expand(self, worst, centroid, gamma=2):
        return centroid + (centroid - worst) * gamma

    def contract(self, worst, centroid, rho=-.5):
        return centroid + (centroid - worst) * rho

    def reduce(self, best, other, sigma=.5):
        return best + (other - best) * sigma

    def compute_next(self, vertices, vertex_ranks):

        worst_index = np.argmax(vertex_ranks)
        worst = vertices[worst_index]
        cent = self.centroid(vertices)

        reflection = self.reflect(worst, cent)
        expansion = self.expand(worst, cent)
        contraction = self.contract(worst, cent)
        return reflection, expansion, contraction, worst_index

    def update_points(self, vertices, rank_func=query_ranks):

        # Get ranks of the current vertex
        vertex_ranks = rank_func(vertices)
        reflection, expansion, contraction, worst_index =\
            self.compute_next(vertices, vertex_ranks)

        # Fetch ranks for all possible next points
        points = np.array(vertices.tolist() + [reflection, expansion, contraction])
        ranks = rank_func(points)
        updated_vertex_ranks = ranks[:len(vertices)]
        ref_rank, exp_rank, con_rank = ranks[len(vertices):]

        mut_type = self.decide_mutation(updated_vertex_ranks, ref_rank, exp_rank, con_rank)
        if mut_type in [Mutation.EXPAND, Mutation.REFLECT, Mutation.CONTRACT]:
            if mut_type == Mutation.EXPAND:
                new_point = expansion
            elif mut_type == Mutation.REFLECT:
                new_point = reflection
            elif mut_type == Mutation.CONTRACT:
                new_point = contraction
            vertices[worst_index] = new_point
        else:
            best_index = np.argmin(vertex_ranks)
            best = vertices[best_index]
            for i, vertex in enumerate(vertices):
                if i != best_index:
                    vertices[i] = self.reduce(best, vertex)

        return vertices

    def decide_mutation(self, vertex_ranks, ref_rank, exp_rank, con_rank):

        if ref_rank < np.min(vertex_ranks) and exp_rank < ref_rank:
            return Mutation.EXPAND
        elif ref_rank < np.sort(vertex_ranks)[-2]:
            return Mutation.REFLECT
        elif con_rank < np.max(vertex_ranks):
            return Mutation.CONTRACT
        else:
            return Mutation.REDUCE

    def optimize(self, x0, rank_func=query_ranks, callback=None):

        vertices = x0
        last_vertices = np.array([])
        vertices_changed = True

        while vertices_changed:

            vertices = self.update_points(vertices, rank_func)
            if callback is not None:
                callback(vertices)

            vertices_changed = np.any(vertices != last_vertices)
            last_vertices = vertices


class SimplexExecutor(object):
    ''' Object for running individual steps of simplex optimization. '''

    def get_next_points(self, vertices, vertex_ranks):
        simplex = Simplex()
        return simplex.compute_next(vertices, vertex_ranks)[:3]

    def update_points(self, vertices, vertex_ranks, ref_rank, exp_rank, con_rank):

        simplex = Simplex()

        def ranker(points):
            if len(points) == len(vertices):
                return np.array(vertex_ranks)
            elif len(points) > len(vertices):
                return np.array(
                    np.array(vertex_ranks).tolist() +
                    [ref_rank, exp_rank, con_rank]
                )

        updated_vertices = simplex.update_points(vertices, rank_func=ranker)
        return updated_vertices

    def is_finished(self, vertices, *args, **kwargs):
        updated_vertices = self.update_points(vertices, *args, **kwargs)
        return np.all(updated_vertices == vertices)


class Mutation(Enum):
    REFLECT = 1
    EXPAND = 2
    CONTRACT = 3
    REDUCE = 4


def debug(X):
    print "Current points:", X


if __name__ == '__main__':
    simplex = Simplex()
    simplex.optimize(
        np.array([[0, 0], [0, 1], [2, 0]]),
        callback=debug,
    )
