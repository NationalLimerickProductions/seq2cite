"""
Assembling a parsed dataset of citation spans from the CORD-19 data
"""
import csv
import multiprocessing as mp
from collections import namedtuple
from typing import Union

from tqdm import tqdm
import pandas as pd
import numpy as np

from seq2cite import config, utils, aws, text


CHUNK_SIZE = 16
# Number of sentences at or before the inline citation to include as context
CONTEXT_SIZE = 1
ARTICLES_FILE = config.raw / 'cord19_articles.csv'
CITATIONS_FILE = config.raw / 'cord19_context_citations.csv'

ARTICLES_NAMES = ['cord_uid', 'title', 'authors', 'date', 'journal', 'doi']
CITATIONS_NAMES = ['cord_uid', 'context', 'cited_title', 'cited_authors', 'cited_date', 'cited_journal']


def load_metadata(offset=0,
                  chunk_size=CHUNK_SIZE,
                  colnames=config.metadata_columns) -> pd.DataFrame:
    header = None if colnames is None else 0
    df = pd.read_csv(f's3://{config.cord19_aws_bucket}/2020-04-10/metadata.csv',
                     nrows=chunk_size,
                     skiprows=offset,
                     names=colnames,
                     header=header)
    df = df[~pd.isna(df['sha'])]
    return df


def worker(row: namedtuple) -> Union[tuple, None]:

    cord_uid = row['cord_uid']

    sha = row['sha'].split('; ')[0]
    title = row['title']
    date = row['publish_time']
    doi = row['doi']
    authors = row['authors'].split('; ')
    journal = row['journal']
    subset = row['full_text_file']

    jsondict = aws.read_item(subset, sha)
    if jsondict is None:
        return None
    context_citations = get_citation_data(cord_uid, jsondict)

    article_info = (cord_uid, title, authors, date, journal, doi)
    return article_info, context_citations


def process_chunk_mp(chunk: pd.DataFrame) -> list:
    print(f'Multiprocessing with {mp.cpu_count()} workers...')
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
            cord_uid = row.cord_uid

            shas = row.sha.split('; ')
            title = row.title
            date = row.publish_time
            doi = row.doi
            authors = row.authors.split('; ')
            journal = row.journal
            subset = row.full_text_file

            for sha in shas:
                jsondict = aws.read_item(subset, sha)
                if jsondict is None:
                    continue
                context_citations = get_citation_data(cord_uid, jsondict)

                articles.append((cord_uid, title, authors, date, journal, doi))
                citation_data.append(context_citations)

            pbar.update()

    return articles, citation_data


def get_citation_data(cord_uid: str, article: dict) -> list:
    """Get the citation data for a given article (in dict format)

    :param cord_uid: The UID of the article from the CORD-19 database
    :param article: The article to be parsed
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
        sents = list(text_section.sents)
        sent_ends = np.array([sent.end_char for sent in sents])

        # Need to loop through the citation spans and get the `CONTEXT_SIZE`
        # sentences before or including the citation span
        for cite_span in cite_spans:
            ref_id = cite_span['ref_id']
            if not ref_id:
                continue
            bibref = bib_entries[ref_id]
            target = (
                bibref['title'],
                bibref['authors'],
                bibref['year'],
                bibref['venue']
            )

            cite_end = cite_span['end']
            end_sent = np.searchsorted(sent_ends, cite_end, side='left')
            start_sent = max(end_sent - CONTEXT_SIZE, 0)
            context_sents = sents[start_sent:end_sent + 1]

            # Masking citations
            inline_text = cite_span['text']
            context = ' '.join([text.mask_span(sent.text, inline_text, mask='<CITE>') for sent in context_sents])

            # Packaging it all together
            datum = (cord_uid, context) + target
            citation_data.append(datum)

    return citation_data


@utils.time_func
def main():
    offset = 0
    total = 0
    fp_articles = ARTICLES_FILE.open('w')
    fp_citations = CITATIONS_FILE.open('w')

    articles_writer = csv.writer(fp_articles)
    citations_writer = csv.writer(fp_citations)

    articles_writer.writerow(ARTICLES_NAMES)
    citations_writer.writerow(CITATIONS_NAMES)

    # Main loop to process in chunks
    print("Beginning processing")
    while True:
        metadata_chunk = load_metadata(offset, CHUNK_SIZE)
        if len(metadata_chunk) == 0:
            break

        all_data = process_chunk_mp(metadata_chunk)

        articles = []
        citation_data = []
        for elem in all_data:
            if elem is None:
                continue
            articles.append(elem[0])
            citation_data.append(elem[1])

        articles_writer.writerows(articles)
        citations_writer.writerows(citation_data)

        total += len(articles)
        print(f'Processed {len(articles)} articles; Total articles processed: {total}')

        offset += CHUNK_SIZE

    print(f"Done. Processed {total} total articles.")
    fp_articles.close()
    fp_citations.close()


if __name__ == '__main__':
    main()
