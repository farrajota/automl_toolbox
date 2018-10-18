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
from automl_toolbox.exceptions import raise_invalid_task_error, raise_clustering_not_implemented_error
from automl_toolbox.utils import parse_or_infer_task_name, transform_data_target


def profiler(df: pd.DataFrame,
             target: str,
             task: str = None,
             show: str = 'all'
             ) -> Dict[str, Union[ProfileReport, dict]]:
    """Creates a profile report for the input data.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    target : str
        Name of the target feature.
    task : str, optional
        Type of task to analyze. If no task is passed as input,
        it will be inferred using the target label with the input
        DataFrame. Options: 'classification', 'cls', 'regression',
        'reg', 'clustering', 'cluster'. Default: None.
    show : str, optional
        Manages what information of the profile report is displayed
        on screen. Options: 'all', 'full', 'basic'. Default: 'all'.

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
        raise Exception(f"Undefined type for 'show': {type(show)}")

    if show_components in ['all', 'model']:
        task_name = parse_or_infer_task_name(df, target, task)
        profile_model: dict = evaluate_model_on_data(df, target, task_name)
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
    X, y = transform_data_target(df, target)
    cross_val_data: dict = cross_validation_score(X, y, task)
    scores_mean: float = cross_val_data["scores"].mean()
    scores_std: float = cross_val_data["scores"].std()
    cv: int = cross_val_data["cv"]
    metric: str = cross_val_data["metric"]

    num_feats: int = len(X.columns)
    num_samples: int = len(X)
    if task == 'classification':
        html: str = create_html_model_evaluation_classification(
            scores_mean=scores_mean,
            scores_std=scores_std,
            num_feats=num_feats,
            cv=cv,
            metric=metric)
    elif task == 'regression':
        html: str = create_html_model_evaluation_regression(
            scores_mean=scores_mean,
            scores_std=scores_std,
            num_feats=num_feats,
            cv=cv,
            metric=metric)
    elif task == 'clustering':
        raise_clustering_not_implemented_error()
    else:
        raise_invalid_task_error(task)

    return {
        "task": task,
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
                                                metric: str
                                                ) -> str:
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
                                            metric: str
                                            ) -> str:
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
    raise Exception("Functionality not yet implemented.")
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
