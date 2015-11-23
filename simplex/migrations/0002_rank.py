# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('simplex', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Rank',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('ipAddr', models.CharField(max_length=32, null=True)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('index', models.IntegerField()),
                ('rank', models.IntegerField()),
                ('iteration', models.IntegerField()),
                ('type', models.CharField(max_length=32)),
            ],
        ),
    ]
