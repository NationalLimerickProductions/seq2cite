"""
text.py

Module for performing text preprocessing and other NLP tasks
"""
import re
import scispacy
import spacy


nlp = spacy.load("en_core_sci_lg", disable=['ner'])


def get_sentences(text: str) -> list:
    """Get a list of the sentences in `text`

    :param text: text to be broken into sentences
    :return: 'sents'
    """
    doc = nlp(text)
    return list(doc.sents)


def mask_span(text: str, span: str, mask='<CITE>') -> str:
    """Mask all occurences of `span` in `text` by replacing with `mask`

    :param text: Text to be processed
    :param span_to_mask: The specific span in `text` to mask
    :param mask: The mask to replace all occurrences of `span` with
    :return: 'masked'
    """
    return text.replace(span, mask)
