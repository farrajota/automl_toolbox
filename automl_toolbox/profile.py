"""
Methods for creating a profile report of a Pandas DataFrame.
"""


import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas_profiling import ProfileReport
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from typing import Dict, Union, List, Optional


def profiler(df: pd.DataFrame,
             target: str,
             method: str = 'classification',
             evaluate_model: bool = True,
             show_report: bool = True
             ) -> Dict[str, Union[ProfileReport, dict]]:
    """Creates a profile report for the input data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target : str
        Name of the feature target for training a model
    method : str, default = 'classification'
        Name of the task to train the model (e.g., classification, regression)
    evaluate_model : bool, default = True
        Train a model (if True)
    show_report : bool, default = True
        Plot the profile info to the screen

    Returns
    -------
    dict
        Report data.
    """
    profile_data = ProfileReport(df)
    if evaluate_model:
        profile_model = evaluate_model_on_data(df, target, method)
    if show_report:
        display_report(profile_data, profile_model)
    return {
        "report": profile_data,
        "model": profile_model
    }


def display_report(profile_data: ProfileReport,
                   profile_model: Union[dict, None]
                   ) -> None:
    """Displays the profile report of the input data."""
    try:
        get_ipython
        from IPython.core.display import display, HTML
        html_display: str = profile_data.html
        if profile_model:
            split_str: str = '</div>\n    <div class="row headerrow highlight">\n        <h1>Variables</h1>'
            html_filtered: str = profile_data.html.split(split_str)
            html_filtered = html_filtered[0] + profile_model["html"] + '</div>\n</div>'
            html_display = html_filtered
        display(HTML(html_display.replace("Overview", "DataFrame Profile Overview")))
    except NameError:
        display_report_text(profile_data, profile_model)


def display_report_text(profile_data: ProfileReport,
                        profile_model: Union[dict, None]
                        ) -> None:
    """Displays the report info as text on screen."""
    print("Overview\n")
    print("")
    print("Dataset info\n")
    print("Number of variables: {}")
    print("Number of observations: {}")
    print("Total Missing: {}")
    print("Total size in memory: {}")
    print("Average record size in memory: {}")
    print("")
    print("Variables types")
    print("Numeric  {}")
    print("Categorical  {}")
    print("Boolean  {}")
    print("Date  {}")
    print("Text (Unique)  {}")
    print("Rejected  {}")
    print("Unsupported  {}")
    print("")
    print("Warning")
    for warning in warning:
        print("  - {}")
    if profile_model:
        print("Model evaluation")
        print("{method} score: {score} (LightGBM)".format(
            method=profile_model["method"],
            score=profile_model["score"]))


def detect_numerical_low_unique_rows_percentage(df: pd.DataFrame,
                                                thresh_unique: float = 20,
                                                verbose: bool = True
                                                ) -> Dict[str, float]:
    """Checks for low unique number of rows in categorical features."""
    if verbose:
        print('\n==> Analysing numerical features:')
    features_low: Dict[str, float] = {}
    for feature in df:
        if is_numeric_dtype(df[feature]):
            n_unique = df[feature].nunique()
            feature_size = len(df[feature].dropna())
            n_unique_percentage = n_unique / feature_size * 100
            if n_unique_percentage < thresh_unique:
                features_low[feature] = n_unique_percentage
                if verbose:
                    print(f'{feature}: {n_unique_percentage}%  ({n_unique}/{feature_size})')
    return features_low


def evaluate_model_on_data(df: pd.DataFrame,
                           target: str,
                           method: str
                           ) -> Dict[str, Union[str, float]]:
    """Trains and evaluates the performance of a lightgbm model
    on the data."""
    if method.lower() in ['classification', 'cls', 'clf']:
        eval_info: float = evaluate_classifier(df, target)
        html: str = create_html_model_evaluation_classification(eval_info)
        method_: str = 'classification'
    elif method.lower() in ['regression', 'reg']:
        eval_info: float = evaluate_regressor(df, target)
        html: str = create_html_model_evaluation_regression(eval_info)
        method_: str = 'regression'
    else:
        raise Exception(f"Undefined method: {method}.")
    return {
        "method": method_,
        "eval_info": eval_info,
        "html": html
    }


def evaluate_classifier(df: pd.DataFrame,
                        target: Union[str, List[str]],
                        cv: int = 5
                        ) -> Dict[str, Union[float, int]]:
    """Train and evaluate a classifier model on an input data."""
    X = df.drop(columns=[target])
    X = pd.get_dummies(X, drop_first=True)
    y = df[target]
    clf = lgb.LGBMClassifier()
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv)
    return {
        "score": scores.mean(),
        "nfeats": len(X.columns),
        "cross_validation": 5
    }


def evaluate_regressor(df: pd.DataFrame,
                       target: Union[str, List[str]],
                       cv: int = 5
                       ) -> Dict[str, Union[float, int]]:
    """Train and evaluate a regressor model on an input data."""
    X = df.drop(columns=[target])
    X = pd.get_dummies(X, drop_first=True)
    y = df[target]
    reg = lgb.LGBMRegressor()
    scores = cross_val_score(reg, X, y, scoring='neg_mean_squared_error', cv=5)
    return {
        "score": scores.mean(),
        "nfeats": len(X.columns),
        "cross_validation": 5
    }


def create_html_model_evaluation_classification(eval_info: Dict[str, Union[str, float]]) -> str:
    return """
        <div>
            <p class="h4">Model evaluation</p>
            <table class="stats" style="margin-left: 1em;">
                <tbody>
                <tr>
                    <th>Task</th>
                    <td>Classification</td>
                </tr>
                <tr>
                    <th>Model</th>
                    <td>LightGBM</td>
                </tr>
                <tr>
                    <th>Number of cv splits</th>
                    <td>{cv}</td>
                </tr>
                <tr>
                    <th>Number of features</th>
                    <td>{nfeats}</td>
                </tr>
                <tr>
                    <th>Accuracy</th>
                    <td>{score:.2f}%</td>
                </tr>
                </tbody>
            </table>
        </div>
    """.format(cv=eval_info['cross_validation'], nfeats=eval_info['nfeats'], score=eval_info['score'] * 100)


def create_html_model_evaluation_regression(eval_info: Dict[str, Union[str, float]]) -> str:
    return """
        <div>
            <p class="h4">Model evaluation</p>
            <table class="stats" style="margin-left: 1em;">
                <tbody>
                <tr>
                    <th>Task</th>
                    <td>Regression</td>
                </tr>
                <tr>
                    <th>Model</th>
                    <td>LightGBM</td>
                </tr>
                <tr>
                    <th>Number of cv splits</th>
                    <td>{cv}</td>
                </tr>
                <tr>
                    <th>Number of features</th>
                    <td>{nfeats}</td>
                </tr>
                <tr>
                    <th>Score</th>
                    <td>{score:.2f}</td>
                </tr>
                </tbody>
            </table>
        </div>
    """.format(cv=eval_info['cross_validation'], nfeats=eval_info['nfeats'], score=eval_info['score'])
