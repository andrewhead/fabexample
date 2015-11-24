#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import logging
import numpy as np
import json

from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.contrib.staticfiles.templatetags.staticfiles import static
from ipware.ip import get_real_ip

from simplex import SimplexExecutor, Simplex
from models import LoadPageEvent, Job, Rank


logging.basicConfig(level=logging.INFO, format="%(message)s")
simplex_executor = SimplexExecutor()


def home(request):
    exemplar_index = request.GET.get('exemplar', 1)
    LoadPageEvent.objects.create(ipAddr=get_real_ip(request))
    return render(request, 'simplex/home.html', {
        'img': static("simplex/img/exemplar" + str(exemplar_index) + ".png"),
    })


def sliders(request):
    exemplar_no = int(request.GET.get('exemplar', 1))
    return render(request, 'simplex/control.html', {
        'img': static("simplex/img/exemplar" + str(exemplar_no) + ".png"),
    })


def step(request):

    iteration = int(request.GET.get('iteration'))
    points = json.loads(request.GET.get('points'))

    # Store the ranks that have been given so far
    for p in points:
        Rank.objects.create(
            rank=p['rank'],
            index=p['index'],
            iteration=iteration,
            type=p['type'],
        )

    simplex = Simplex()
    new_points = simplex.step(points)

    for p in new_points:
        if 'rank' not in p:
            Job.objects.create(
                ipAddr=get_real_ip(request),
                value=json.dumps(p['value']),
                type=p['type'],
            )

    return JsonResponse({
        'points': new_points,
    })


def update_vertices(request):

    vertices = np.array(json.loads(request.GET.get('vertices')), dtype=np.float)
    orig_vertices = vertices.copy()
    vertex_ranks = np.array([int(_) for _ in request.GET.getlist('vertex_ranks[]')])
    vertex_indices = [int(_) for _ in request.GET.getlist('vertex_indices[]')]
    reflection_rank = int(request.GET.get('reflection_rank'))
    expansion_rank = int(request.GET.get('expansion_rank'))
    contraction_rank = int(request.GET.get('contraction_rank'))
    reflection_index = int(request.GET.get('reflection_index'))
    expansion_index = int(request.GET.get('expansion_index'))
    contraction_index = int(request.GET.get('contraction_index'))
    iteration = int(request.GET.get('iteration'))

    # Store the ranks that have been given so far
    for i in range(len(vertices)):
        Rank.objects.create(
            rank=vertex_ranks[i],
            index=vertex_indices[i],
            iteration=iteration,
            type='vertex',
        )
    Rank.objects.create(
        rank=expansion_rank, index=expansion_index, iteration=iteration, type='expansion')
    Rank.objects.create(
        rank=reflection_rank, index=reflection_index, iteration=iteration, type='reflection')
    Rank.objects.create(
        rank=contraction_rank, index=contraction_index, iteration=iteration, type='contraction')

    updated_vertices = simplex_executor.update_points(
        vertices, vertex_ranks, reflection_rank, expansion_rank, contraction_rank,
    )

    dropped_indices = []
    new_values = []
    for i in range(len(orig_vertices)):
        # This uses default paramters for absolute and relative closeness,
        # though note that if you're working on a simplex with very small
        # numbers (1e-5 or below), you should add new tolerances
        if not np.allclose(orig_vertices[i], updated_vertices[i]):
            Job.objects.create(
                ipAddr=get_real_ip(request),
                value=json.dumps(updated_vertices[i].tolist()),
                type="vertex",
            )
            dropped_indices.append(vertex_indices[i])
            new_values.append(updated_vertices[i].tolist())

    return JsonResponse({
        'new': new_values,
        'dropped': dropped_indices,
    })


def get_next(request):

    vertices = json.loads(request.GET.get('vertices'))
    vertex_ranks = request.GET.getlist('vertex_ranks[]')
    vertex_indices = [int(_) for _ in request.GET.getlist('vertex_indices[]')]
    reflection, expansion, contraction = simplex_executor.get_next_points(
        np.array(vertices, dtype=np.float),
        np.array(vertex_ranks),
    )

    for i in range(len(vertices)):
        Rank.objects.create(
            rank=vertex_ranks[i],
            index=vertex_indices[i],
            iteration=request.GET.get('iteration'),
            type='vertex',
        )

    for point, type_ in zip(
            [reflection, expansion, contraction],
            ['reflection', 'expansion', 'contraction']):
        Job.objects.create(
            ipAddr=get_real_ip(request),
            value=json.dumps(point.tolist()),
            type=type_,
        )

    return JsonResponse({
        'reflection': reflection.tolist(),
        'expansion': expansion.tolist(),
        'contraction': contraction.tolist(),
    })


def submit_job(request):
    values = [float(_) for _ in request.GET.getlist('values[]')]
    Job.objects.create(
        ipAddr=get_real_ip(request),
        value=json.dumps(values),
        type='manual',
    )
    return HttpResponse()


def queue(request):

    jobs = Job.objects.all().order_by('-timestamp')[:20]
    jobs_augmented = []
    for j in jobs:
        value = json.loads(j.value)
        if type(value) == list:
            if len(value) == 3:
                digits = (1, 0, 0)
                j.value = json.dumps(
                    [round(v, digits[i]) for i, v in enumerate(value)]
                )
                j.exp_value = json.dumps(
                    [round(np.power(v, 3.162), digits[i])
                        for i, v in enumerate(value)]
                )
                jobs_augmented.append(j)

    return render(request, 'simplex/queue.html', {
        'jobs': jobs_augmented,
    })
