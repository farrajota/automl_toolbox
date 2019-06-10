# AutoML toolbox - PROJECT DEPRECATED

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

The **Auto**matic **M**achine **L**earning (AutoML) toolbox is a collection of methods with a simple API which can assist you in your Data Science tasks by providing boilerplate code in a form of simple functions ready to be used. You can use it as your personal Data Science assistant / wizard for creating ETL processes and/or a fully automated machine learning pipelines in a simple and quick way.

This library is intended to be used as a testing playground for a bunch of wrapper methods to serve as high-level APIs to a bunch of common tasks like:

- data profiling
- cleaning missing values
- detecting outliers
- performing feature engineering
- hyper-parameter optimization
- evaluating machine learning models
- creating ensembles of such models
- etc.

## Warning

This code base is in heavy development for now. Once it reaches `v0.1.0` you may then try it, but for now you are at your own risk.

## Installation

For now, to install this package you must build it from source. To do that, just run the following command in the terminal:

```bash
python setup.py install
```

> Note: once this package reaches `v0.1.0` it will be possible to install it via pip.

## Key Libraries used

This toolbox integrates the following packages in its core for doing most of its work. Basically, you can think of this package as a wrapper for a bunch functions you would uneed like cross-validation, hyperparameter optimization, etc., but with a nice, high-level API.

- Numpy
- Pandas
- pandas-profiling (data profiler)
- Scikit-learn (collection of ML libs)
- xgboost (ML lib)
- lightgbm (ML lib)
- Hyperopt (hyperparam optim - bo)
- HpBandSter (hyperparam optim - hyperband + bo)

### Libraries to be integrated in the future

- dask (distributed computing / big data)
- keras (DL lib)
- feature-tools (automatic feature engineering)
- [pygdf](https://github.com/rapidsai/pygdf) (GPU DataFrame)

## TODO

Funcionalities intended to be added to the toolbox:

- [x] basic data profiler
- [ ] automatic analysis / benchmarking and filling of missing values
- [ ] automatic analysis / benchmarking and cleaning of outliers
- [ ] automatic feature transformations / normalization
- [ ] automatic feature engineering
- [ ] automatic feature selection
- [ ] automatic model selection
- [ ] automatic model optimization (hyper-parameter optimization)
- [ ] automatic model ensembling
- [ ] pre-defined parameter list of the most popular ML models in scikit-learn
- [ ] distributed computing (integrate Dask)
- [ ] pipeline generation

## License

[MIT](LICENSE)
