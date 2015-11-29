#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import logging
import numpy as np
import scipy.stats


logging.basicConfig(level=logging.INFO, format="%(message)s")


'''
In We will have these methods for predicting the value at a point:
Method | Given | Return
get_f_map | f0[n, 1], Hessian method, gradient method | f[n, 1]
get_distinct_x | comparisons[m, 2] | x[n, 1]
kernel | x1, x2 | value
kernel_vector | kernel(), x[n, 1], xnew | products[n, 1]
kernel_matrix | kernel(), x[n, 1] | products[n, n]
C | f[n, 1], comparisons[m, 2], x[n, 1], sigma_noise | C[n, n]
H | f[n, 1], comparisons[m, 2], x[n, 1], sigma_noise | H[n, n]
b | f[n, 1], comparisons[m, 2], x[n, 1], sigma_noise | b[n, 1]
g | kernel(), f[n, 1], comparisons[m, 2], x[n, 1], sigma_noise | g[n, 1]
predict | kernel(), get_f_map(), f0 | y

And these methods for optimizing the selection of the next point:
sigma | kernel(), f[n, 1], comparisons[m, 2], x[n, 1], sigma_noise | variance
expected_improvement | kernel(), f[n, 1], comparisons[m, 2], x[n, 1], sigma_noise | ei
next_point | kernel(), f[n, 1], comparisons[m, 2], x[n, 1], sigma_noise | x
'''


N = scipy.stats.norm()


def newton_rhapson(x, f0, comparisons, kernel, Hfunc, gfunc, sigma, maxiter=100):
    f = f0
    i = 0
    while i < maxiter:
        H = Hfunc(kernel, x, f, comparisons, sigma)
        Hinv = np.linalg.inv(H)
        g = gfunc(kernel, x, f, comparisons, sigma)
        step = Hinv.dot(g)
        f = f - step
        print "f (", i, ")", f
        i += 1
    return f


def c_pdf_cdf_term(z):
    phi = N.pdf(z)
    Phi = N.cdf(z)
    return (phi / np.power(Phi, 2)) + (np.power(phi, 2) / Phi) * z


def compute_z(fr, fc, sigma):
    return (fr - fc) / (np.sqrt(2.0) * sigma)


def h(comparison, xi):
    r, c = comparison
    return \
        0.0 if xi == r and xi == c else \
        1.0 if xi == r else \
        -1.0 if xi == c else \
        0.0


def b_summand(f, j, comparison, sigma):
    ri, ci = comparison
    fr = f[ri][0]
    fc = f[ci][0]
    z = compute_z(fr, fc, sigma)
    hi = h(comparison, j)
    summand = (N.pdf(z) / N.cdf(z)) * hi
    return summand


def b_j(f, j, comparisons, sigma):
    sum_ = 0
    num_comp = comparisons.shape[0]
    for ci in range(num_comp):
        comp = comparisons[ci]
        summand = b_summand(f, j, comp, sigma)
        sum_ += summand
    b = sum_ / (np.sqrt(2) * sigma)
    return b


def compute_b(f, comparisons, sigma):
    # We assume we have the same number of 'f' as we do of 'x'
    b = []
    for fi in range(len(f)):
        b_row = [b_j(f, fi, comparisons, sigma)]
        b.append(b_row)
    return np.array(b)


def compute_g(kernel, x, f, comparisons, sigma):
    K = kernel_matrix(kernel, x)
    Kinv = np.linalg.inv(K)
    b = compute_b(f, comparisons, sigma)
    g = (-1 * Kinv.dot(f)) + b
    return g


def c_summand(f, m, n, comparison, sigma):
    '''
    Params:
    f: vector of evaluations for all points x
    m: index of one point
    n: index of another point
    comparison: two indices, each for one of a pair of points
    sigma: standard deviation of the noise
    '''
    ri, ci = comparison
    fr = f[ri][0]
    fc = f[ci][0]
    z = compute_z(fr, fc, sigma)
    cdf_pdf_term = c_pdf_cdf_term(z)
    hm = h(comparison, m)
    hn = h(comparison, n)
    return hm * hn * cdf_pdf_term


def c_m_n(f, m, n, comparisons, sigma):
    sum_ = 0
    num_comp = comparisons.shape[0]
    for ci in range(num_comp):
        comp = comparisons[ci]
        summand = c_summand(f, m, n, comp, sigma)
        sum_ += summand
    c = sum_ / (2 * sigma * sigma)
    return c


def compute_C(f, comparisons, sigma):
    # We assume we have the same number of 'f' as we do of 'x'
    C = []
    for fi in range(len(f)):
        c_row = []
        for fj in range(len(f)):
            c_entry = c_m_n(f, fi, fj, comparisons, sigma)
            c_row.append(c_entry)
        C.append(c_row)
    C_np = np.array(C)
    return C_np


def compute_H(kernel, x, f, comparisons, sigma):
    K = kernel_matrix(kernel, x)
    Kinv = np.linalg.inv(K)
    C = compute_C(f, comparisons, sigma)
    H = Kinv + C
    return H


def get_distinct_x(comparisons):
    xlist = []
    in_list = lambda l, e: len(filter(lambda x: np.allclose(x, e), l)) > 0
    for r, c in comparisons:
        if not in_list(xlist, r):
            xlist.append(r)
        if not in_list(xlist, c):
            xlist.append(c)
    return np.array(xlist)


def get_comparison_indices(x, comparisons):
    indices = []
    row_count = comparisons.shape[0]
    for rowi in range(row_count):
        r, c = comparisons[rowi]
        ri = -1
        ci = -1
        for xi, xe in enumerate(x):
            if np.allclose(r, xe):
                ri = xi
            if np.allclose(c, xe):
                ci = xi
        indices.append([ri, ci])
    return np.array(indices)


def default_kernel(x1, x2):
    diff = x1 - x2
    return np.exp(-.5 * diff.dot(diff))


def kernel_vector(kernel, x, xnew):
    klist = []
    for xe in x:
        klist += [[kernel(xe, xnew)]]
    return np.array(klist)


def kernel_matrix(kernel, x):
    K = []
    for xe1 in x:
        row = []
        for xe2 in x:
            row.append(kernel(xe1, xe2))
        K.append(row)
    return np.array(K)
