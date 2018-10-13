"""
Data cleaning methods and functions. With this module, users can:

- profile their data (see missing values, type of fields, their distribution, etc.)
- automatically clean missing values with different methods at their disposal (mean, mode, imputation, etc.)
- automatically detect and / or clean outliers and profile their impact on the data
"""

from ._profile import profiler
