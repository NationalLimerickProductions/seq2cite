import json
from time import time
from operator import itemgetter
import ast
from pathlib import PosixPath, Path
from typing import Union
import csv

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


def keep_nlargest(dct: dict, n: int) -> dict:
    """Return a dict with only the n-largest elements in `dct` (by value)

    Parameters
    ----------
    dct: dict
    n: int.

    Returns
    -------
    'n_largest': dict.
    """
    res = dict(sorted(dct.items(), key=itemgetter(1), reverse=True)[:n])
    return res


def load_vocab(vocab_file: Union[PosixPath, str]) -> dict:
    """Load a vocab file from JSON

    Example usage:

        author_vocab = load_vocab(config.final / 'author_vocab.json')

    Parameters
    ----------
    vocab_file: The absolute or relative path to the file.

    Returns
    -------
    'vocab'
    """
    vocab_file = _check_path(vocab_file)
    with vocab_file.open('r') as f:
        dct = json.load(f)
    return dct


def load_data(data_file: Union[PosixPath, str], nrows=None) -> list:
    """Load the data file (context and citations), preprocessing as necessary

    Example usage:

        data = load_data(config.final / 'cord19_data_clean.csv')


    Parameters
    ----------
    data_file
    nrows

    Returns
    -------

    """
    data_file = _check_path(data_file)
    data = []
    with data_file.open('r') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='"')
        for i, row in enumerate(csvreader):
            if nrows is not None:
                if i >= nrows:
                    break
            rowdata = []
            for j, elem in enumerate(row):
                if j > 0:
                    elem = ast.literal_eval(elem)
                rowdata.append(elem)
            data.append(rowdata)
    return data


def _check_path(path):
    if type(path) == str:
        path = Path(path)
    if type(path) != PosixPath:
        raise TypeError(f'Type of path is {type(path)}. Must be either str or PosixPath.')
    return path
