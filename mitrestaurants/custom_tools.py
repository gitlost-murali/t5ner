import sys

def find_tags_andiob(searchphrases_withtags, sent):
    """
    Given a set of prediction phrases with their tags
    And a sentence,
    Now, where the phrases fit in and return tag list for that sentence
    Args:
        searchphrases ([type]): [description]
        sent ([type]): [description]
        tags ([type]): [description]

    Returns:
        tags [List]: [A list of tags in iob format]
    """
    tags = ['O' for _ in sent]
    for searchphrase, tag in searchphrases_withtags:
        if searchphrase=="": continue
        searchtokens = searchphrase.split()
        if len(searchtokens)==0: continue
        phraselen = len(searchtokens)
        iobseq = [f"I-{tag}" for _ in range(phraselen)]
        try:
            iobseq[0] = f"B-{tag}"
        except Exception as e:
            print(f"Error occured IOB seq {iobseq}")
            print(f"Error occured searchtokens -> {searchtokens}")
            print(f"Error occured searchphrases -> {searchphrase}")
            print(f"Error occured searchphrases -> {searchphrases_withtags}")
            print(f"Error occured sent -> {sent}")
            sys.exit(0)

        for ix, _ in enumerate(sent):
            if searchtokens == sent[ix:ix+phraselen]:
                tags[ix:ix+phraselen] = iobseq

    assert len(sent) == len(tags)
    return tags

# Testing

# sample = [["is", "the", "a", "chau", "restaurant", "within", "a", "mile", "from", "here", "a", "local", "favorite"],
# [ "O", "O", "O", "B-Restaurant_Name", "O", "B-Location", "I-Location", "I-Location", "O", "O", "O", "O", "O" ]]

# searchphrase = "chau restaurant"
# convert_tags([searchphrase], sample[0],["GF"])


from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

import json

def calculate_seqeval(y_true, y_pred):
    fscore = f1_score(y_true, y_pred)
    cm = classification_report(y_true, y_pred)
    return fscore, cm