#!/usr/bin/env python

from setuptools import setup

setup(name='dsn',
      version='1.0',
      description='Degenerate solution networks.',
      author='Sean Bittner',
      author_email='srb2201@columbia.edu',
      packages=['dsn', 'dsn.util'],
      install_requires=['tf_util', 'tensorflow', 'numpy', 'statsmodels', \
                        'scipy', 'cvxopt', 'matplotlib', 'scikit-learn'],
      dependency_links=['https://github.com/cunningham-lab/tf_util/tarball/master#egg=tf_util-1.0'],
     )
