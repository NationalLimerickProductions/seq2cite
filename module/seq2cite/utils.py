from IPython.display import display
import pandas as pd
import boto3

from . import config


def connect_aws_s3():
    """Open a connection to AWS S3.
    """
    s3 = boto3.client('s3')
    s3_resource = boto3.resource('s3')
    return s3, s3_resource


def get_cord19_bucket(s3=None, s3_resource=None):
    if s3 is None and s3_resource is None:
        s3, s3_resource = connect_aws_s3()
    assert s3 is not None and s3_resource is not None, \
        'Must provide both or neither of s3 and s3_resource'

    return s3_resource.Bucket(config.cord19_aws_bucket)


def display_all_rows(df):
    with pd.option_context("display.max_rows", None):
        display(df)


def display_all_cols(df):
    with pd.option_context("display.max_columns", None):
        display(df)


def display_all(df):
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        display(df)
