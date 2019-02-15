#!/usr/bin/env python

from setuptools import setup

setup(name='dsn',
      version='0.1',
      description='Degenerate solution networks.',
      author='Sean Bittner',
      author_email='srb2201@columbia.edu',
      dependency_links= ['git+https://github.com/cunningham-lab/tf_util.git@master#egg=tf_util-0.1'],
      install_requires=['tf_util==0.1', 'numpy', 'scipy', 'tensorflow', \
                        'statsmodels', 'matplotlib', 'scikit-learn'],
      packages=['dsn', 'dsn.util'],
     )
