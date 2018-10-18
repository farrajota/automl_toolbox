"""
The Data Science Assistant
==========================

The Data Science ``Assistant`` provides a collection of methods to address
the most typical procedures when analyzing data. Such processes include::

- Profiling data
- Filling missing values
- Detecting and removing outliers
- Feature transformations
- Feature engineering
- Feature selection
- Model selection
- Model hyper-parameter optimization
- Ensemble

This assistant can help you automate or reduce many of the boiler plate code
or repetitive tasks for dealing with Data Science projects or ETL processes.

Through a simple, but configurable API, you are able to achieve the bulk of
the work in data analysis scenarios with some few methods. Furthermore,
detailed reports for each procedure are generated and made available to the user
so he/she can analyze the results of the full process and gather valuable insights
on potential improvements with little to no effort.

The ``Assistant`` class wraps the full functionality of ``automl_toolbox``
into a single object with lost of methods to work with and it enables users
to access and modify its configurations and set it up according to their needs.
"""


import pandas as pd
from pandas_profiling import ProfileReport
from typing import Dict, Union, List, Optional

from .data_cleaning import profiler
from .utils import parse_backend_name_string, parse_or_infer_task_name


class Assistant(object):
    """Data Science Assistance / Wizard.

        Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    target : str
        Target column name.
    task : str, optional
        Type of task to analyze. If no task is passed as input,
        it will be inferred using the target label with the input
        DataFrame. Options: 'classification', 'cls', 'regression',
        'reg', 'clustering', 'cluster'. Default: None.
    backend : str, optional
        Name of the model's backend. Default: 'lightgbm'.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 target: str,
                 task: str = None,
                 backend: str = 'lightgbm'
                 ) -> None:
        self.df = df
        self.target = target
        self.task = parse_or_infer_task_name(self.df, self.target, task)
        self.backend = parse_backend_name_string(backend)
        self.data = DataProfiler(self.df, self.target, self.task, self.backend)


class DataProfiler(object):
    """Analyses and profiles the data.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    target : str
        Target column name.
    task : str, optional
        Type of task to analyze. If no task is passed as input,
        it will be inferred using the target label with the input
        DataFrame. Options: 'classification', 'cls', 'regression',
        'reg', 'clustering', 'cluster'. Default: None.
    backend : str, optional
        Name of the model's backend. Default: 'lightgbm'.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 target: str,
                 task: str = None,
                 backend: str = 'lightgbm') -> None:
        self.df = df
        self.target = target
        self.task = parse_or_infer_task_name(self.df, self.target, task)
        self.backend = parse_backend_name_string(backend)

    def profile(self,
                df: pd.DataFrame = None,
                target: str = None,
                task: str = None,
                show: Union[bool, str, list] = 'all'
                ) -> Dict[str, Union[ProfileReport, dict]]:
        """Generates profile reports from a Pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            Input pandas DataFrame to be profiled. Default: None.
        target : str, optional
            Target column of the DataFrame. Default: None.
        task : str, optional
            Type of task to analyze. If no task is passed as input,
            it will be inferred using the target label with the input
            DataFrame. Options: 'classification', 'cls', 'regression',
            'reg', 'clustering', 'cluster'. Default: None.
        show : bool | str | list, optional
            Manages what information of the profile report is displayed
            on screen. Options: 'all', 'full', 'basic'. Default: 'all'

        Returns
        -------
        dict
            Report of the data.
        """
        if df:
            df_analysis = df
            target_analysis = target
            task_analysis = parse_or_infer_task_name(df_analysis, target_analysis, task)
        else:
            df_analysis = self.df
            target_analysis = self.target
            task_analysis = self.task
        report: dict = profiler(df=df_analysis,
                                target=target_analysis,
                                task=task_analysis,
                                show=show)
        return report
