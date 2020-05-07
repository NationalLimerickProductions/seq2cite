"""
Assembling a parsed dataset of citation spans from the CORD-19 data
"""
import csv
import multiprocessing as mp
from collections import namedtuple
from typing import Union
import pickle

from tqdm import tqdm
import pandas as pd
import numpy as np

from seq2cite import config, utils, aws, text


CHUNK_SIZE = 256
# Number of tokens surrounding the citation to take as context
CONTEXT_SIZE = 30
ARTICLES_FILE = config.raw / 'cord19_articles.csv'
CITATIONS_FILE = config.raw / 'cord19_context_citations.csv'
AUTHOR_VOCAB_FILE = config.raw / 'cord19_author_vocab.pkl'
CITE_TOKEN = '<CITE>'
CITE_IDX = text.nlp.vocab.strings[CITE_TOKEN]
ARTICLES_NAMES = ['cord_uid', 'title', 'authors', 'date', 'journal', 'doi']
CITATIONS_NAMES = ['citation_id', 'context', 'auth_idxs', 'citing_auth_idxs', 'title_idxs']
MIN_DATE = pd.to_datetime('2005-01-01')


author_vocab = mp.Manager().dict()
# author_vocab = dict()

def load_metadata(offset=0,
                  chunk_size=CHUNK_SIZE,
                  colnames=config.metadata_columns
                  ) -> Union[pd.DataFrame, None]:
    header = None if colnames is None else 0
    df = pd.read_csv(f's3://{config.cord19_aws_bucket}/2020-04-10/metadata.csv',
                     nrows=chunk_size,
                     skiprows=offset,
                     names=colnames,
                     header=header)
    if len(df) == 0:
        return None
    df = df[~pd.isna(df['sha'])]
    df = df[pd.to_datetime(df['publish_time']) > MIN_DATE]
    return df


def worker(row: namedtuple) -> Union[tuple, None]:
    global author_vocab

    cord_uid = row['cord_uid']

    sha = row['sha'].split('; ')[0]
    title = row['title']
    date = row['publish_time']
    doi = row['doi']
    journal = row['journal']
    subset = row['full_text_file']

    jsondict = aws.read_item(subset, sha)
    if jsondict is None:
        return
    authors = jsondict['metadata'].get('authors')
    auth_idxs = get_author_idxs(authors)
    context_citations = get_citation_data(cord_uid, jsondict, auth_idxs)

    article_data = (cord_uid, title, auth_idxs, date, journal, doi)

    return article_data, context_citations


def process_chunk_mp(chunk: pd.DataFrame) -> list:
    print(f'Multiprocessing {len(chunk)} records with {mp.cpu_count()} workers...')
    data = chunk.to_dict('records')
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(worker, data)

    return results


def process_chunk(chunk: pd.DataFrame) -> tuple:
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

    2. Returns a tuple of articles, citation_data, with the following formats

        articles:

            (cord_uid, title, authors, date, journal, doi)

        citation_data:

            (cord_uid, context, (cited_title, cited_authors, cited_date, cited_journal))

    #TODO: Use multiprocessing

    :param chunk: Chunk of the metadata
    :return: 'articles', 'citation_data'
    """
    articles = []
    citation_data = []
    with tqdm(total=len(chunk)) as pbar:
        for row in chunk.itertuples():
            cord_uid = row['cord_uid']

            shas = row.sha.split('; ')
            title = row['title']
            date = row['publish_time']
            doi = row['doi']
            journal = row['journal']
            subset = row['full_text_file']

            for sha in shas:
                jsondict = aws.read_item(subset, sha)
                if jsondict is None:
                    continue
                authors = jsondict['metadata'].get('authors')
                auth_idxs = get_author_idxs(authors)
                context_citations = get_citation_data(cord_uid, jsondict, auth_idxs)

                articles.append((cord_uid, title, auth_idxs, date, journal, doi))
                citation_data.extend(context_citations)

                # Just using the first sha
                break

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
            title_idxs = [t.lemma for t in text.nlp(title)]

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
            context_pre = list([t.idx for t in text_section[idx_start - context_size_pre:idx_start]])
            context_post = list([t.idx for t in text_section[idx_end:idx_end + context_size_post]])
            context = context_pre + [CITE_IDX] + context_post

            # Packaging it all together
            citation_id = f'{cord_uid}__{ref_id}'
            datum = (citation_id, context, auth_idxs, citing_auth_idxs, title_idxs)
            citation_data.append(datum)

    return citation_data


@utils.time_func
def main():
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
    print("Beginning processing")
    try:
        while True:
            metadata_chunk = load_metadata(offset, CHUNK_SIZE)
            chunk_idx += 1
            if metadata_chunk is None:
                break
            if len(metadata_chunk) == 0:
                print(f'Skipping chunk {chunk_idx} with length 0')
                continue
            print(f'Processing chunk {chunk_idx}')

            # all_data = process_chunk(metadata_chunk)
            # articles, citation_data = all_data

            all_data = process_chunk_mp(metadata_chunk)
            articles = []
            citation_data = []
            for elem in all_data:
                if elem is None:
                    continue
                articles.append(elem[0])
                citation_data.extend(elem[1])

            articles_writer.writerows(articles)
            citations_writer.writerows(citation_data)

            total_articles += len(articles)
            total_citations += len(citation_data)
            print(f'Processed {len(articles)} articles; Total articles processed: {total_articles}; Total citations processed: {total_citations}')

            offset += CHUNK_SIZE
    except KeyboardInterrupt:
        with AUTHOR_VOCAB_FILE.open('wb') as f:
            pickle.dump(author_vocab, f)
    finally:
        print(f"Done. Processed {total_articles} total articles with {total_citations} citations.")
        fp_articles.close()
        fp_citations.close()


if __name__ == '__main__':
    main()
