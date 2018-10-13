"""
Pre-defined list of hyper-parameters of LightGBM.
"""


parameters = {
    # Core Parameters
    "boosting": {
        "type": 'str',
        "default": 'gbdt',
        "sampling": 'choice',
        "values": ['gbdt', 'rf', 'dart'],
        "sklearn_alias": 'boosting_type'
    },
    "num_iterations": {
        "type": 'int',
        "default": 100,
        "sampling": 'uniform',
        "min_range": 10,  # num_iterations >= 0
        "max_range": 1000
    },
    "learning_rate": {
        "type": 'float',
        "default": 0.1,
        "sampling": 'loguniform',
        "min_range": 0.001,
        "max_range": 10
    },
    "num_leaves": {
        "type": 'int',
        "default": 31,
        "sampling": 'uniform',
        "min_range": 2,  # num_leaves > 1
        "max_range": 500
    },
    "tree_learner": {
        "type": 'str',
        "default": 'serial',
        "sampling": 'choice',
        "values": ['serial', 'feature', 'data', 'voting']
    },
    # Learning Control Parameters
    "min_data_in_leaf": {
        "type": 'int',
        "default": 20,
        "sampling": 'loguniform',
        "min_range": 1,  # min_data_in_leaf >= 0
        "max_range": 500
    },
    "min_sum_hessian_in_leaf": {
        "type": 'float',
        "default": 1e-3,
        "sampling": 'uniform',
        "min_range": 1e-6,  # min_sum_hessian_in_leaf >= 0.0
        "max_range": 1
    },
    "bagging_fraction": {
        "type": 'float',
        "default": 1.0,
        "sampling": 'uniform',
        "min_range": 1e-3,  # 0.0 < bagging_fraction <= 1.0
        "max_range": 1
    },
    "bagging_freq": {
        "type": 'float',
        "default": 0,
        "sampling": 'uniform',
        "min_range": 0,  # 0.0 <= bagging_freq <= 1.0
        "max_range": 1
    },
    "feature_fraction": {
        "type": 'float',
        "default": 1.0,
        "sampling": 'uniform',
        "min_range": 1e-6,  # 0.0 <= bagging_freq <= 1.0
        "max_range": 1
    }
}


def get_default_parameters() -> dict:
    default_parameters = {}
    for key in parameters:
        default_parameters[key] = parameters[key]['default']
    return default_parameters
