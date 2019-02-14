#!/usr/bin/env python

from setuptools import setup

setup(name='dsn',
      version='0.1',
      description='Degenerate solution networks.',
      author='Sean Bittner',
      author_email='srb2201@columbia.edu',
      packages=['dsn', 'dsn.util'],
      install_requires=['numpy', 'scipy', 'tensorflow', \
                        'statsmodels', 'matplotlib', 'scikit-learn'],
     )
