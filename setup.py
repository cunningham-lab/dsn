#!/usr/bin/env python

from setuptools import setup

setup(name='dsn',
      version='0.1',
      description='Degenerate solution networks.',
      author='Sean Bittner',
      author_email='srb2201@columbia.edu',
      install_requires=['numpy', 'tensorflow==2.9.3', \
                        'statsmodels', 'matplotlib', 'scikit-learn', 'scipy'],
      packages=['dsn', 'dsn.util'],
     )
