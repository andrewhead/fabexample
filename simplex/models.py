#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import logging
from django.db import models


logging.basicConfig(level=logging.INFO, format="%(message)s")


class Rank(models.Model):
    ipAddr = models.CharField(max_length=32, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    index = models.IntegerField()
    rank = models.IntegerField()
    iteration = models.IntegerField()
    type = models.CharField(max_length=32)


class LoadPageEvent(models.Model):
    ipAddr = models.CharField(max_length=32, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)


class Job(models.Model):
    ipAddr = models.CharField(max_length=32, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    value = models.CharField(max_length=128)
    type = models.CharField(max_length=32)
