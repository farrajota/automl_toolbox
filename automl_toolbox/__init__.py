"""
AutoML Toolbox
==============

The Automatic Machine Learning toolbox (AutoML) is a collection of methods
with a simple API which can assist you in your Data Science tasks by providing
boilerplate code in a form of simple functions ready to be used. You can use
it as your personal Data Science assistant / wizard for creating ETL processes
and/or a fully automated machine learning pipelines in a simple and quick way.

This library is intended to be used as a testing playground for a bunch of
wrapper methods to serve as high-level APIs to a bunch of common tasks like:

- data profiling
- cleaning missing values
- detecting outliers
- performing feature engineering
- hyper-parameter optimization
- evaluating machine learning models
- creating ensembles of such models
- etc.
"""


import pkg_resources

# package version
__version__ = pkg_resources.get_distribution('automl_toolbox').version
