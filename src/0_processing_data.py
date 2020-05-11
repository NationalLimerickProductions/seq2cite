"""
Assembling a parsed dataset of citation spans from the CORD-19 data
"""
import csv
import json
import sys
import multiprocessing as mp
from collections import namedtuple
from typing import Union
import pickle
import tarfile
from io import BytesIO

from tqdm import tqdm
import pandas as pd
import numpy as np
import boto3

from seq2cite import config, utils, aws, text


# Number of tokens surrounding the citation to take as context
CONTEXT_SIZE = 30
ARTICLES_FILE = config.processed / 'cord19_articles.csv'
CITATIONS_FILE = config.processed / 'cord19_context_citations.csv'
AUTHOR_VOCAB_FILE = config.processed / 'cord19_author_vocab.json'
TOKEN_VOCAB_FILE = config.processed / 'cord19_token_vocab.json'
TITLE_VOCAB_FILE = config.processed / 'cord19_title_vocab.json'
CITE_TOKEN = '<CITE>'
CITE_IDX = 1
ARTICLES_NAMES = ['cord_uid', 'title', 'authors', 'date', 'journal', 'doi']
CITATIONS_NAMES = ['citation_id', 'context', 'auth_idxs', 'citing_auth_idxs', 'title_idxs']
MIN_DATE = pd.to_datetime('2010-01-01')
KEYS = {'arxiv': '',
        'noncomm_use_subset': '2020-04-10',
        'biorxiv_medrxiv': '2020-04-17',
        'custom_license': '2020-04-10',
        'comm_use_subset': '2020-04-10'}

author_vocab = {'<UNK>': 0, '<PAD>': 9999999}
token_vocab = {'<UNK>': 0, '<CITE>': 1, '<PAD>': 9999999}
title_vocab = {'<UNK>': 0, '<CITE>': 1, '<PAD>': 9999999}
curr_author_idx = 1
curr_token_idx = 2
curr_title_idx = 2


def load_metadata(offset=0,
                  chunk_size=None,
                  colnames=config.metadata_columns
                  ) -> Union[pd.DataFrame, None]:
    header = None if colnames is None else 0
    df = pd.read_csv(config.raw / 'metadata.csv',
                     nrows=chunk_size,
                     skiprows=offset,
                     names=colnames,
                     header=header,
                     index_col=False)
    if len(df) == 0:
        return None
    df = df[~pd.isna(df['sha'])]
    df = df[pd.to_datetime(df['publish_time']) > MIN_DATE]
    return df


def load_tar_files():
    res = {}
    for key, date_ in KEYS.items():
        print(f'Loading {key}')
        keyfile = config.raw / f'{key}.tar.gz'
        try:
            content = tarfile.open(keyfile, mode="r:gz")
        except tarfile.ReadError:
            content = tarfile.open(keyfile)
        members = content.getmembers()
        member_dict = {m.name.split('/')[-1].rstrip('.json'): m for m in members}
        res[key] = {}
        res[key]['tarfile'] = content
        res[key]['members'] = member_dict
    return res


def get_tarfile(tarfiles, subset, sha):
    tar_subset = tarfiles.get(subset, None)
    if tar_subset is None:
        return
    content = tarfiles[subset]['tarfile']
    member = tarfiles[subset]['members'][sha]
    return json.load(content.extractfile(member))


def process_chunk(chunk: pd.DataFrame, tarfiles: dict) -> tuple:
    """Steps in processing a chunk

    1. For each article:
        a) Extract metadata:
            - cord_uid
            - Title
            - Authors
            - Date
            - Journal
            - DOI
        c) Load JSON article
        d) Extract all citation spans. Convert each inline citation to a tuple:

            (cord_uid, context, (cited_title, cited_authors, cited_date, cited_journal))

    2. Returns a tuple of articles, citation_data

    :param chunk: Chunk of the metadata
    :return: 'articles', 'citation_data'
    """
    articles = []
    citation_data = []
    with tqdm(total=len(chunk)) as pbar:
        for row in chunk.itertuples():
            cord_uid = row.cord_uid

            sha = row.sha.split('; ')[0]
            title = row.title
            date = row.publish_time
            doi = row.doi
            journal = row.journal
            subset = row.url

            jsondict = get_tarfile(tarfiles, subset, sha)
            if jsondict is None:
                continue
            authors = jsondict['metadata'].get('authors')
            auth_idxs = get_author_idxs(authors)
            context_citations = get_citation_data(cord_uid, jsondict, auth_idxs)

            articles.append((cord_uid, title, auth_idxs, date, journal, doi))
            citation_data.extend(context_citations)

            pbar.update()

    return articles, citation_data


def get_author_idxs(authors: list) -> list:
    """Return a list of author idxs for the authors in `authors`.

    Adds the authors to the vocab in the process.

    :param authors: List of authors (list[str])
    :return: 'auth_idxs' (list[int])
    """
    global author_vocab

    auth_idxs = []
    for author in authors:
        auth_abbrev = f'{author["first"][:1]} {author["last"]}'
        if auth_abbrev in author_vocab:
            auth_idx = author_vocab[auth_abbrev]
        else:
            auth_idx = len(author_vocab)
            author_vocab[auth_abbrev] = auth_idx
        auth_idxs.append(auth_idx)
    return auth_idxs


def get_token_idx(token: str) -> int:
    """Get the token id for a given token, adding it to the vocab if necessary.

    Parameters
    ----------
    token {str} -- Token to add to the vocab

    Returns
    -------
    'id' {int}
    """
    global curr_token_idx, token_vocab

    if token not in token_vocab:
        token_vocab[token] = curr_token_idx
        curr_token_idx += 1
    return token_vocab[token]


def get_title_idx(token: str) -> int:
    """Get the token id for a given title token, adding it to the vocab
    if necessary.

    Parameters
    ----------
    token {str} -- Token to add to the vocab

    Returns
    -------
    'id' {int}
    """
    global curr_title_idx, title_vocab

    if token not in title_vocab:
        title_vocab[token] = curr_title_idx
        curr_title_idx += 1
    return title_vocab[token]


def get_citation_data(cord_uid: str, article: dict, citing_auth_idxs: list) -> list:
    """Get the citation data for a given article (in dict format)

    :param cord_uid: The UID of the article from the CORD-19 database
    :param article: The article to be parsed
    :param citing_auth_idxs: The citing authors (encoded)
    :return: 'citation_data', a list of tuples:

        (cord_uid, context, (cited_title, cited_authors, cited_date, cited_journal))
    """
    bib_entries = article['bib_entries']
    body_text = article['body_text']
    citation_data = []
    text_sections = [section['text'] for section in body_text]
    text_sections = list(text.nlp.pipe(text_sections))
    for section, text_section in zip(body_text, text_sections):
        cite_spans = section['cite_spans']
        sent_len = len(text_section)

        # Need to loop through the citation spans and get the `CONTEXT_SIZE`
        # sentences before or including the citation span
        for cite_span in cite_spans:
            ref_id = cite_span['ref_id']
            if not ref_id:
                continue
            bibref = bib_entries[ref_id]
            authors = bibref['authors']
            auth_idxs = get_author_idxs(authors)
            title = bibref['title']
            title_idxs = [get_title_idx(t.lemma_) for t in text.nlp(title)]

            # Finding the context
            cite_start = cite_span['start']
            cite_end = cite_span['end']
            idx_start, idx_end = 0, sent_len
            for t in text_section:
                if t.idx == cite_start:
                    idx_start = t.i
                if t.idx + len(t) == cite_end:
                    idx_end = t.i

            # Fitting the window into the section
            context_size_pre, context_size_post = CONTEXT_SIZE / 2, CONTEXT_SIZE / 2
            if idx_start < context_size_pre:
                context_size_post += idx_start - context_size_pre
                context_size_pre = idx_start
            if sent_len - idx_end < context_size_post:
                context_size_post = sent_len - idx_end

            # Getting the context
            context_pre = list([get_token_idx(t.lemma_) for t in text_section[idx_start - context_size_pre:idx_start]])
            context_post = list([get_token_idx(t.lemma_) for t in text_section[idx_end:idx_end + context_size_post]])
            context = context_pre + [CITE_IDX] + context_post

            # Packaging it all together
            citation_id = f'{cord_uid}__{ref_id}'
            datum = (citation_id, context, auth_idxs, citing_auth_idxs, title_idxs)
            citation_data.append(datum)

    return citation_data


@utils.time_func
def main():
    CHUNK_SIZE = 10000
    offset = 0
    total_articles = 0
    total_citations = 0
    chunk_idx = -1
    fp_articles = ARTICLES_FILE.open('w')
    fp_citations = CITATIONS_FILE.open('w')

    articles_writer = csv.writer(fp_articles)
    citations_writer = csv.writer(fp_citations)

    articles_writer.writerow(ARTICLES_NAMES)
    citations_writer.writerow(CITATIONS_NAMES)

    # Main loop to process in chunks
    print("Loading TAR files")
    tarfiles = load_tar_files()
    print("Beginning processing")
    try:
        while True:
            metadata_chunk = load_metadata(offset, chunk_size=CHUNK_SIZE)
            chunk_idx += 1
            if metadata_chunk is None:
                break
            if len(metadata_chunk) == 0:
                # print(f'Skipping chunk {chunk_idx} with length 0')
                continue
            print(f'Processing chunk {chunk_idx}')
            all_data = process_chunk(metadata_chunk, tarfiles)
            articles, citation_data = all_data

            articles_writer.writerows(articles)
            citations_writer.writerows(citation_data)

            total_articles += len(articles)
            total_citations += len(citation_data)
            print(f'Processed {len(articles)} articles; Total articles processed: {total_articles}; Total citations processed: {total_citations}')

            offset += CHUNK_SIZE
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Done. Processed {total_articles} total articles with {total_citations} citations.")
        fp_articles.close()
        fp_citations.close()

        for vocab, file in zip((author_vocab, token_vocab, title_vocab),
                               (AUTHOR_VOCAB_FILE, TOKEN_VOCAB_FILE, TITLE_VOCAB_FILE)):
            idx2vocab = {v: k for k, v in vocab.items()}
            with file.open('w') as f:
                json.dump(idx2vocab, f)


if __name__ == '__main__':
    main()
