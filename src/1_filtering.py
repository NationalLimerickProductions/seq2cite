"""
Filtering authors and tokens to keep only the 20k most common of each. Uncommon
authors/tokens are set to UNK_author or UNK.
"""
import csv
import json
from operator import itemgetter
import ast

import pandas as pd
import numpy as np
from tqdm import tqdm, trange

from seq2cite import config, utils


UNK = 0
CITE = 1
PAD_TOKEN = 9999999
N_TO_KEEP = 20000


def load_data(subset=None) -> tuple:
    """
    Returns
    -------
    'citations_context', 'author_vocab'
    """
    citations_context = pd.read_csv(config.processed / 'cord19_context_citations.csv', nrows=subset)
    for col in ["context", "auth_idxs", "title_idxs"]:
        citations_context[col] = citations_context[col].apply(ast.literal_eval)
    token_vocab = json.load((config.processed / 'cord19_token_vocab.json').open('r'))
    title_vocab = json.load((config.processed / 'cord19_title_vocab.json').open('r'))
    author_vocab = json.load((config.processed / 'cord19_author_vocab.json').open('r'))
    return citations_context, token_vocab, title_vocab, author_vocab


def count_occurrences(list_of_seqs: list) -> dict:
    """

    Parameters
    ----------
    list_of_seqs {iterable} -- A list of sequences of tokens (token ids)

    Returns
    -------
    token_counts {dict} -- Dictionary mapping each token id to the number of occurrences in the corpus
    """
    token_counts = {}
    for seq in tqdm(list_of_seqs):
        for token in seq:
            n = token_counts.get(token, 0)
            token_counts[token] = n + 1
    return token_counts


def prepare_vocab(counts: dict, vocab: dict, include_cite=True) -> tuple:
    """Convert the existing author vocab to two dicts: one mapping the original
    IDs to the new IDs and the other mapping new IDs to the author strings.

    Parameters
    ----------
    counts: {dict} Counts of each author (keys are IDs)
    vocab: {dict} Dict mapping author IDs to strings
    include_cite: {bool} Whether to include the <CITE> token in the new vocab

    Returns
    -------
    'vocab_new', 'replacement_dict'
    """
    vocab_new, replacement_dict = {UNK: '<UNK>', PAD_TOKEN: '<PAD>'}, {}
    if include_cite:
        vocab_new[CITE] = '<CITE>'

    vocab_largest = utils.keep_nlargest(counts, N_TO_KEEP)
    for i, idx_old in tqdm(list(enumerate(vocab_largest))):
        # Account for UNK and CITE, which is idx 0
        idx_new = i + 2
        replacement_dict[idx_old] = idx_new
        string = vocab[f"{idx_old}"]
        vocab_new[idx_new] = string
    return replacement_dict, vocab_new


def apply_vocab(vocab_replacement_dict: dict, list_of_seqs: list) -> list:
    """Apply the new vocab replacement to the sequences in `list_of_seqs`.
    Any tokens that are not in the vocab are assigned to <UNK> (index 0)

    WARNING: OPERATES IN-PLACE

    Parameters
    ----------
    vocab_replacement_dict: {dict} Dictionary mapping from old IDs to new IDs
    list_of_seqs: {list} List of sequences to apply the new vocab to.

    Returns
    -------
    'replaced_seqs' {list}
    """
    res = []
    for seq in tqdm(list_of_seqs):
        new_seq = [vocab_replacement_dict.get(idx, UNK) for idx in seq]
        res.append(new_seq)
    return res


@utils.time_func
def main():
    print("Loading data")
    citations_context, token_vocab, title_vocab, author_vocab = load_data(subset=1000)

    print("Getting token counts")
    token_counts = count_occurrences(citations_context['context'].tolist())
    print("Getting title counts")
    title_counts = count_occurrences(citations_context['title_idxs'].tolist())
    print("Getting author counts")
    author_counts = count_occurrences(citations_context['auth_idxs'].tolist())

    print("Preparing token vocab")
    token_replacement_dict, token_vocab_new = prepare_vocab(token_counts, token_vocab)
    print("Preparing title vocab")
    title_replacement_dict, title_vocab_new = prepare_vocab(title_counts, title_vocab)
    print("Preparing author vocab")
    author_replacement_dict, author_vocab_new = prepare_vocab(author_counts, author_vocab, include_cite=False)

    print("Applying vocab to contexts")
    contexts_replaced = apply_vocab(token_replacement_dict, citations_context['context'].tolist())
    print("Applying vocab to titles")
    titles_replaced = apply_vocab(title_replacement_dict, citations_context['title_idxs'].tolist())
    print("Applying author vocab to authors")
    authors_replaced = apply_vocab(author_replacement_dict, citations_context['auth_idxs'].tolist())

    citations_context['context'] = contexts_replaced
    citations_context['title_idxs'] = titles_replaced
    citations_context['auth_idxs'] = authors_replaced

    print("Saving")
    citations_context.to_csv(config.final / 'cord19_data_clean.csv', header=False)
    json.dump(token_vocab_new, (config.final / 'token_vocab.json').open('w'))
    json.dump(title_vocab_new, (config.final / 'title_vocab.json').open('w'))
    json.dump(author_vocab_new, (config.final / 'author_vocab.json').open('w'))
    print("Done")

if __name__ == '__main__':
    main()
