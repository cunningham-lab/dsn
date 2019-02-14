#!/usr/bin/env python

from setuptools import setup

setup(name='dsn',
      version='1.0',
      description='Degenerate solution networks.',
      author='Sean Bittner',
      author_email='srb2201@columbia.edu',
      packages=['dsn', 'dsn.util'],
      install_requires=['numpy', 'statsmodels', \
                        'scipy', 'matplotlib', 'scikit-learn'],
     )
