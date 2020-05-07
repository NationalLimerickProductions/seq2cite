import json
from typing import Union

import requests
import pandas as pd
import boto3
from botocore.exceptions import ClientError, ReadTimeoutError

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


def read_item(subset: str, id_: str, date='2020-04-17', s3=None) -> Union[dict, None]:
    """Read a single article in JSON format

    :param subset: The collection the article belongs to
    :param id_: The ID of the article
    :param date: The date (optional)
    :param s3: S3 client (optional)
    :return: 'jsondict'
    """
    if s3 is None:
        s3 = boto3.client('s3')
    key = f'{date}/{subset}/pdf_json/{id_}.json'
    try:
        result = s3.get_object(Bucket=config.cord19_aws_bucket,
                               Key=key)
        jsondict = json.loads(result['Body'].read().decode())
        return jsondict
    except json.JSONDecodeError:
        print(f"Error parsing file {key}")
        return None
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f'No such key: {key}')
        else:
            raise
    except ReadTimeoutError:
        print(f'Error reading file {key}')
        return None
