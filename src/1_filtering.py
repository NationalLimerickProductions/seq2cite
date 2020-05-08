"""
Filtering authors and tokens to keep only the 20k most common of each. Uncommon
authors/tokens are set to UNK_author or UNK.
"""
import csv
import json

import pandas as pd
import numpy as np
from tqdm import tqdm
import spacy

from seq2cite import config, utils, text


UNK_TOKEN = 0
UNK_AUTHOR = 0
PAD_TOKEN = 9999999


def load_data() -> tuple:
    """
    Returns
    -------
    'citations_context', 'author_vocab'
    """
    citations_context = pd.read_csv(config.processed / 'cord19_context_citations.csv')
    author_vocab = json.load(config.processed / 'cord19_author_vocab.json')
    return citations_context, author_vocab


def count_occurrences(list_of_seqs: list) -> dict:
    """

    Parameters
    ----------
    list_of_seqs {iterable} -- A list of sequences of tokens (token ids)

    Returns
    -------
    token_counts {dict} -- Dictionary mapping each token id to the number of occurrences in the corpus
    """
    pass


def prepare_token_vocab(token_counts: dict, nlp: spacy.language.Language) -> dict:
    """Convert the spacy vocab to two dicts: one mapping the original IDs
    to the new IDs (in the truncated vocab) and the other mapping new IDs to
    the strings

    Parameters
    ----------
    token_counts: {dict} Counts of each token (keys are IDs)
    nlp: {spacy.language.Language} spaCy language object with a vocab as an
        attribute

    Returns
    -------
    'token_vocab_new', 'token_replacement_dict'
    """
    pass


def prepare_author_vocab(author_counts: dict, author_vocab: dict) -> tuple:
    """Convert the existing author vocab to two dicts: one mapping the original
    IDs to the new IDs and the other mapping new IDs to the author strings.

    Parameters
    ----------
    author_counts: {dict} Counts of each author (keys are IDs)
    author_vocab: {dict} Dict mapping author IDs to strings

    Returns
    -------
    'author_vocab_new', 'author_replacement_dict'
    """
    pass


def apply_vocab(vocab_replacement_dict: dict, list_of_seqs: list) -> list:
    """Apply the new vocab replacement to the sequences in `list_of_seqs`.
    Any tokens that are not in the vocab are assigned to <UNK> (index 0)

    Parameters
    ----------
    vocab_replacement_dict: {dict} Dictionary mapping from old IDs to new IDs
    list_of_seqs: {list} List of sequences to apply the new vocab to.

    Returns
    -------
    'replaced_seqs' {list}
    """
    pass


@utils.time_func
def main():
    citations_context, author_vocab = load_data()
    nlp = text.nlp

    contexts_and_titles = citations_context['context'].tolist() + citations_context['title_idxs'].tolist()
    authors = citations_context['auth_idxs'].tolist()

    token_counts = count_occurrences(contexts_and_titles)
    author_counts = count_occurrences(authors)

    del contexts_and_titles
    del authors

    token_vocab_new, token_replacement_dict = prepare_token_vocab(token_counts, nlp)
    author_vocab_new, author_replacement_dict = prepare_author_vocab(author_counts, author_vocab)

    contexts_replaced = apply_vocab(token_replacement_dict, citations_context['context'].tolist())
    titles_replaced = apply_vocab(token_replacement_dict, citations_context['title_idxs'].tolist())
    authors_replaced = apply_vocab(author_replacement_dict, citations_context['auth_idxs'].tolist())

    citations_context['context'] = contexts_replaced
    citations_context['title_idxs'] = titles_replaced
    citations_context['auth_idxs'] = authors_replaced

    citations_context.to_csv(config.final / 'cord19_data_clean.csv', header=False)


if __name__ == '__main__':
    main()
