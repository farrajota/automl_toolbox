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

from automl_toolbox.model_selection import cross_validation_score
from automl_toolbox.exceptions import raise_invalid_task_error, UndefinedMethodError
from automl_toolbox.utils import parse_task_name_string


def profiler(df: pd.DataFrame,
             target: str,
             method: str = 'classification',
             show: str = 'all'
             ) -> Dict[str, Union[ProfileReport, dict]]:
    """Creates a profile report for the input data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target : str
        Name of the feature target for training a model
    method : str, optional (default='classification')
        Name of the task to train the model (e.g., classification, regression)
    show : str, optional (default='all')
        Displays sections of the profile report.

    Returns
    -------
    dict
        Report data.

    TODO
    ----
    - split refactor into multiple sections that can be appended when needed
    - enable selecting only a subset of the variables for display
    """
    report: dict = {}
    profile_data = ProfileReport(df)
    report["report"] = profile_data

    if isinstance(show, bool):
        if show:
            show_components: str = 'all'
        else:
            show_components: str = ''
    elif isinstance(show, str):
        show_components: str = show.lower()
    else:
        raise Exception(f"Undefined type for 'show': {type(bool)}")

    if show_components in ['all', 'model']:
        profile_model: dict = evaluate_model_on_data(df, target, method)
        report["model"] = profile_model
    if show_components:
        display_report(profile_data, profile_model)
    return report


def evaluate_model_on_data(df: pd.DataFrame,
                           target: str,
                           task: str
                           ) -> Dict[str, Union[str, float, int, dict]]:
    """Trains and evaluates the performance of a lightgbm model
    on the data."""
    task_parsed: str = parse_task_name_string(task)

    # Setup data
    X = df.drop(columns=target)
    X = pd.get_dummies(X, drop_first=True)
    y = df[target]

    cross_val_data: dict = cross_validation_score(X, y, task_parsed)
    scores_mean: float = cross_val_data["scores"].mean()
    scores_std: float = cross_val_data["scores"].std()
    cv: int = cross_val_data["cv"]
    metric: str = cross_val_data["metric"]

    num_feats: int = len(X.columns)
    num_samples: int = len(X)
    if task_parsed == 'classification':
        html: str = create_html_model_evaluation_classification(
            scores_mean=scores_mean,
            scores_std=scores_std,
            num_feats=num_feats,
            cv=cv,
            metric=metric)
    elif task_parsed == 'regression':
        html: str = create_html_model_evaluation_regression(
            scores_mean=scores_mean,
            scores_std=scores_std,
            num_feats=num_feats,
            cv=cv,
            metric=metric)
    else:
        raise_invalid_task_error(task_parsed)

    return {
        "task": task_parsed,
        "scores": {
            "mean": scores_mean,
            "std": scores_std,
            "cv": cross_val_data["cv"],
            "metric": metric
        },
        "num_samples": num_samples,
        "num_feats": num_feats,
        "html": html
    }


def create_html_model_evaluation_classification(scores_mean: float,
                                                scores_std: float,
                                                num_feats: int,
                                                cv: int,
                                                metric: str) -> str:
    return f"""
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
                    <td>{num_feats}</td>
                </tr>
                <tr>
                    <th>Metric</th>
                    <td>{metric}</td>
                </tr>
                <tr>
                    <th>Score</th>
                    <td>{scores_mean * 100:.2f}%</td>
                </tr>
                <tr>
                    <th>std</th>
                    <td>{scores_std * 100:.2f}%</td>
                </tr>
                </tbody>
            </table>
        </div>
    """


def create_html_model_evaluation_regression(scores_mean: float,
                                            scores_std: float,
                                            num_feats: int,
                                            cv: int,
                                            metric: str) -> str:
    return f"""
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
                    <td>{num_feats}</td>
                </tr>
                <tr>
                    <th>Metric</th>
                    <td>{metric}</td>
                </tr>
                <tr>
                    <th>Score</th>
                    <td>{scores_mean:.2f}</td>
                </tr>
                <tr>
                    <th>std</th>
                    <td>{scores_std:.2f}</td>
                </tr>
                </tbody>
            </table>
        </div>
    """


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
