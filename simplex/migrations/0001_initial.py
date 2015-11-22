# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Job',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('ipAddr', models.CharField(max_length=32, null=True)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('value', models.CharField(max_length=128)),
                ('type', models.CharField(max_length=32)),
            ],
        ),
        migrations.CreateModel(
            name='LoadPageEvent',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('ipAddr', models.CharField(max_length=32, null=True)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
