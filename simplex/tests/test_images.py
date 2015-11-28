#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import logging
import unittest

from simplex.views import get_cut_image_name


logging.basicConfig(level=logging.INFO, format="%(message)s")


class GetImageForConfigurationTest(unittest.TestCase):

    def test_get_middle_cut(self):
        name = get_cut_image_name(
            power=2,
            speed=2,
            ppi=2,
        )
        self.assertEqual(name, '/static/simplex/img/cuts/100ppi_12.png')

    def test_index_includes_two_digits_even_if_under_ten(self):
        name = get_cut_image_name(
            power=1,
            speed=2,
            ppi=2,
        )
        self.assertEqual(name, '/static/simplex/img/cuts/100ppi_07.png')

    def test_round_power_up(self):
        name = get_cut_image_name(
            power=2.6,
            speed=2,
            ppi=2,
        )
        self.assertEqual(name, '/static/simplex/img/cuts/100ppi_17.png')

    def test_round_power_down(self):
        name = get_cut_image_name(
            power=2.4,
            speed=2,
            ppi=2,
        )
        self.assertEqual(name, '/static/simplex/img/cuts/100ppi_12.png')

    def test_round_all(self):
        name = get_cut_image_name(
            power=2.6,
            speed=2.6,
            ppi=2.6,
        )
        self.assertEqual(name, '/static/simplex/img/cuts/316ppi_18.png')
