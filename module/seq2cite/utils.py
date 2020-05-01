from time import time

from IPython.display import display
import pandas as pd
import boto3

from . import config


def display_all_rows(df):
    with pd.option_context("display.max_rows", None):
        display(df)


def display_all_cols(df):
    with pd.option_context("display.max_columns", None):
        display(df)


def display_all(df):
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        display(df)


def time_func(func):
    """Convenience function to decorate with a timer

    :param func: callable
    :return: callable
    """
    def f(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        end = time()
        print(f'Time elapsed: {end - start} seconds.')
        return res
    return f
