#!/usr/bin/env python
# coding: utf-8

# In[2]:

# ## Reference
# 
# 1.Q: Are the hf checkpoints trained with multi-tasking?
# 
# A: yes -> https://discuss.huggingface.co/t/t5-finetuning-tips/684/9
# 
# 2.Q: From the author on NER
# A. https://github.com/google-research/text-to-text-transfer-transformer/issues/27#issuecomment-569088540
# 
# 3.Q. More on details -> https://github.com/google-research/text-to-text-transfer-transformer/issues/339

# ### Checking a sample on translation
# 
# Reference: https://huggingface.co/transformers/model_doc/t5.html#inference

# ### Prepare a toy dataset for Tuning

# In[3]:


from textwrap import indent
from unittest import skip
from tqdm import tqdm

'''
!pip install datasets
'''

from datasets import load_dataset
dataset = load_dataset('conll2003')
ner_tags = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
ner_id2tags = dict((v,k) for (k,v) in ner_tags.items())

# In[4]:

import config
import pandas as pd
import random
random.seed(config.params["RANDOM_STATE"])
from random import shuffle

import sys
# config.params["TEMPLATE"] = 1 or 2
filter_limit = int(sys.argv[1])

tag_filters = {
"ORG": filter_limit,
"PER": filter_limit,
"LOC": filter_limit,
"MISC": filter_limit
}


# config.params["TEMPLATE"] = 1 or 2
train_dataset = []
dev_dataset = []
testing_dataset = []

for instance in dataset["train"]:
    train_dataset.append((instance['tokens'], [ner_id2tags[tg] for tg in instance['ner_tags']]))

for instance in dataset["validation"]:
    dev_dataset.append((instance['tokens'], [ner_id2tags[tg] for tg in instance['ner_tags']]))

for instance in dataset["test"]:
    testing_dataset.append((instance['tokens'], [ner_id2tags[tg] for tg in instance['ner_tags']]))

# In[8]:


tag_descriptor = {
"ORG": "Organization",
"PER": "Person",
"LOC": "Location",
"MISC": "Miscellaneous Entity"
}

# ### Function to convert IOB tags to normal entity texts with their tags
# 
# ```
# """
# Given a list of lists [w1, w2, w3, w4, w5] & [B-, O, B-,I-, O]
# Return entity texts -> ["w1", "w3 w4"] with ["ORG", "LOC"]
# """
# ```

# In[9]:


def get_entities(tokens,tags, template_type):
    """
    Given a list of lists [w1, w2, w3, w4, w5] & [B-, O, B-,I-, O]
    Return entity texts -> ["w1", "w3 w4"] with ["ORG", "LOC"]
    """
    bucket = []
    tags_bucket = []
    
    tok_idx = []
    
    merged_tokens = []
    merged_tags = []
    merged_tok_idces = []
    
    prevtag = ''
    for idx, (tok, tag) in enumerate(zip(tokens,tags)):
        if tag == "O" and len(bucket) != 0:
            merged_tokens.append(" ".join(bucket))
            merged_tags.append(prevtag)
            merged_tok_idces.append(tok_idx)
            bucket = []
            tok_idx = []
        
        elif tag == "O" and len(bucket) == 0:
            bucket = []
            tok_idx = []
            
        elif tag.startswith("I-"):
            bucket.append(tok)
            tok_idx.append(idx)

        elif tag.startswith("B") and len(bucket) != 0:
            merged_tokens.append(" ".join(bucket))
            merged_tags.append(prevtag)
            bucket = []
            bucket.append(tok)
            tok_idx = []
            tok_idx.append(idx)

        elif tag.startswith("B") and len(bucket) == 0:
            bucket.append(tok)
            tok_idx.append(idx)
        
        prevtag = tag[2:]

    if len(bucket)!=0: 
        merged_tokens.append((" ").join(bucket))
        merged_tags.append(prevtag)
        merged_tok_idces.append(tok_idx)
    
    "Convert tags to descriptions"
    if template_type == 1:
        merged_tags = [tag_descriptor[tg] for tg in merged_tags]

    return merged_tokens, merged_tags, merged_tok_idces


## Count tags
def counttags(taglist):
    tagdict=dict()
    for tg in taglist:
        tagdict[tg] = tagdict.get(tg,0) + 1
    return tagdict

# ### Function to convert ner samples to Templates for model inputs

# ### Function to convert ner samples to Templates for model inputs
def templatize_function(sentence_text, entity_text, entity_tags, template_type):
    """
    Sentence Text: Normal textual sentence
    Entity Text: ['London', 'Iraq', 'British'], 
    Entity Tags: ['Geographical Entity', 'Geographical Entity', 'Geopolitical Entity']
    """

    if template_type == 1:
        target_template = ""
        input_template = f"ner: {sentence_text}"
        for ent, tg in zip(entity_text, entity_tags):
            target_template += f"{ent} [{tg}] "

        target_template = target_template.strip()
        target_template = [target_template]
        input_template = [input_template]

    if template_type == 2:
        input_template = []
        target_template = []
        ents_per_tok = dict()
        ## Create a bucket that stores entities per tag
        for tag, tok in zip(entity_tags, entity_text):
            ents_per_tok[tag] = ents_per_tok.get(tag, []) + [tok]

        tagnames = []
        for tagname, dscr in tag_descriptor.items():
            input_template.append(f"ner: {sentence_text}, the '{dscr}' entities in the sentence are ")
            if tagname in entity_tags:
                target_template.append(f"{ ', '.join(ents_per_tok.get(tagname, 'None')) }")
            else:
                target_template.append("None")

            tagnames.append(tagname)

    return input_template, target_template, tagnames



# In[11]:

shuffle(train_dataset)
shuffle(dev_dataset)
shuffle(testing_dataset)

dataset = []
skipped = 0
considered = 0


import sys

if filter_limit != -1:

    tag_2_sentences = dict()

    for i in tqdm(range(0, len(train_dataset))):
        ans = get_entities(train_dataset[i][0], train_dataset[i][1], template_type= config.params["TEMPLATE"])
        tagsdist = counttags(ans[1])
        for eachtg in tagsdist:
            tag_2_sentences[eachtg] = tag_2_sentences.get(eachtg, []) + [i]

    import json
    # with open("storetag2sent.json",'w') as fh: json.dump(tag_2_sentences, fh, indent=4)

    already_visited = []
    filtered_training = []

    for eachtag in tag_2_sentences:
        # config.logger.info(tag_filters)
        for sntix in tag_2_sentences[eachtag]:
            if tag_filters[eachtag] <= 0: break
            if sntix not in already_visited:
                skipsent = False
                ans = get_entities(train_dataset[sntix][0], train_dataset[sntix][1], template_type= config.params["TEMPLATE"])
                tagsdist = counttags(ans[1])
                for etg in tagsdist:
                    if tag_filters[etg] - tagsdist[etg] < 0:
                        # config.logger.info(f"Skipped {eachtag} in {sntix} bcoz of {etg}")
                        skipsent = True
                        break
                    tag_filters[etg] -= tagsdist[etg]
                    # config.logger.info(f"main tag{eachtag} sentence sub {etg} {tagsdist}")
                    # config.logger.info(f"{tag_filters}")

                
                if skipsent==True: continue
                already_visited.append(sntix)
                filtered_training.append(train_dataset[sntix])

    config.logger.info(f"Length is {len(filtered_training)}, {len(tag_filters.keys())}")
    # with open("filtered.json",'w') as fh: json.dump(filtered_training, fh, indent=4)


    for i in tqdm(range(0, len(filtered_training))):
        ans = get_entities(filtered_training[i][0], filtered_training[i][1], template_type= config.params["TEMPLATE"])

        tagsdist = counttags(ans[1])
        sent_inclusion_flag = True

        
        data_instances = templatize_function(sentence_text = " ".join(filtered_training[i][0]), entity_text = ans[0], \
                        entity_tags = ans[1], template_type=config.params["TEMPLATE"])
        
        for srcinst, tgtinst, tagname in zip(data_instances[0], data_instances[1], data_instances[2]):
            dataset.append([tagname, srcinst, tgtinst])

    config.logger.info(f"Considered {len(filtered_training)} sentences")
    config.logger.info(f"Total sents are {len(train_dataset)}")

# In[12]:
else:
    for i in tqdm(range(0, len(train_dataset))):
        ans = get_entities(train_dataset[i][0], train_dataset[i][1], template_type= config.params["TEMPLATE"])

        data_instances = templatize_function(sentence_text = " ".join(train_dataset[i][0]), entity_text = ans[0], \
                        entity_tags = ans[1], template_type=config.params["TEMPLATE"])

        # print(data_instances[:5])
        for srcinst, tgtinst, tagname in zip(data_instances[0], data_instances[1], data_instances[2]):
            dataset.append([tagname, srcinst, tgtinst])

import json


# In[13]:


with open(f"{config.params['TRAINING_FILE'].replace('.json','')}_temp{config.params['TEMPLATE']}.json", "w") as fh:
    json.dump(dataset, fh, indent=4)


val_dataset = []
for i in tqdm(range(0, len(dev_dataset))):
    ans = get_entities(dev_dataset[i][0], dev_dataset[i][1], template_type= config.params["TEMPLATE"])

    data_instances = templatize_function(sentence_text = " ".join(dev_dataset[i][0]), entity_text = ans[0], \
                    entity_tags = ans[1], template_type=config.params["TEMPLATE"])
    
    for srcinst, tgtinst, tagname in zip(data_instances[0], data_instances[1], data_instances[2]):
        val_dataset.append([tagname, srcinst, tgtinst])

with open(f"{config.params['DEV_FILE'].replace('.json','')}_temp{config.params['TEMPLATE']}.json", "w") as fh:
    json.dump(val_dataset, fh, indent=4)


test_dataset = []
for i in tqdm(range(0, len(testing_dataset))):
    ans = get_entities(testing_dataset[i][0], testing_dataset[i][1], template_type= config.params["TEMPLATE"])

    data_instances = templatize_function(sentence_text = " ".join(testing_dataset[i][0]), entity_text = ans[0], \
                    entity_tags = ans[1], template_type=config.params["TEMPLATE"])
    
    for srcinst, tgtinst, tagname in zip(data_instances[0], data_instances[1], data_instances[2]):
        test_dataset.append([tagname, srcinst, tgtinst])
    
with open(f"{config.params['TEST_FILE'].replace('.json','')}_temp{config.params['TEMPLATE']}.json", "w") as fh:
    json.dump(test_dataset, fh, indent=4)


# ### Some other references
# 
# 1. https://github.com/Shivanandroy/T5-Finetuning-PyTorch
