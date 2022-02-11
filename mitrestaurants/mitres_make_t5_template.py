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

trainpath = "../../restaurant-mit/restauranttrain.bio"
testpath = "../../restaurant-mit/restauranttest.bio"

# In[4]:

import config as mit_rest_config
import pandas as pd
import random
random.seed(mit_rest_config.params["RANDOM_STATE"])
from random import shuffle

import sys
# mit_rest_config.params["TEMPLATE"] = 1 or 2
filter_limit = int(sys.argv[1])

tag_filters = {
"Rating": filter_limit,
"Amenity": filter_limit,
"Location": filter_limit,
"Restaurant_Name": filter_limit,
"Price": filter_limit,
"Hours": filter_limit,
"Dish":filter_limit,
"Cuisine": filter_limit,
}


with open(trainpath,'r') as fh: traindf = fh.read()
with open(testpath,'r') as fh: testingdf = fh.read()

# mit_rest_config.logger.info(traindf.head(10))
# In[7]:
training_dataset = []
sent_text = []
sent_tags = []
for row in traindf.split("\n"):
    try:
        tag, word = row.split("\t")
    except:
        word = "nan"
    if str(word)=="nan": 
        training_dataset.append([sent_text, sent_tags])
        sent_text = []
        sent_tags = []
    else:
        sent_text.append(word)
        sent_tags.append(tag)

testing_dataset = []
testing_sent_text = []
testing_sent_tags = []

for row in testingdf.split("\n"):
    try:
        tag, word = row.split("\t")
    except:
        word = "nan"
    if str(word)=="nan": 
        testing_dataset.append([testing_sent_text, testing_sent_tags])
        testing_sent_text = []
        testing_sent_tags = []
    else:
        testing_sent_text.append(word)
        testing_sent_tags.append(tag)

# In[8]:


tag_descriptor = {
"Rating": "Rating",
"Amenity": "Amenity",
"Location": "Location",
"Restaurant_Name": "Restaurant Name",
"Price": "Price",
"Hours": "Hours",
"Dish":"Dish",
"Cuisine": "Cuisine",
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

shuffle(training_dataset)
training_dataset = training_dataset[:-int(mit_rest_config.params["VALIDATION_SPLIT"] * len(training_dataset))]
dev_dataset = training_dataset[-int(mit_rest_config.params["VALIDATION_SPLIT"] * len(training_dataset)):]
shuffle(testing_dataset)

dataset = []
skipped = 0
considered = 0


import sys


tag_2_sentences = dict()

for i in tqdm(range(0, len(training_dataset))):
    ans = get_entities(training_dataset[i][0], training_dataset[i][1], template_type= mit_rest_config.params["TEMPLATE"])
    tagsdist = counttags(ans[1])
    for eachtg in tagsdist:
        tag_2_sentences[eachtg] = tag_2_sentences.get(eachtg, []) + [i]

import json
# with open("storetag2sent.json",'w') as fh: json.dump(tag_2_sentences, fh, indent=4)

already_visited = []
filtered_training = []

for eachtag in tag_2_sentences:
    # mit_rest_config.logger.info(tag_filters)
    for sntix in tag_2_sentences[eachtag]:
        if tag_filters[eachtag] <= 0: break
        if sntix not in already_visited:
            skipsent = False
            ans = get_entities(training_dataset[sntix][0], training_dataset[sntix][1], template_type= mit_rest_config.params["TEMPLATE"])
            tagsdist = counttags(ans[1])
            for etg in tagsdist:
                if tag_filters[etg] - tagsdist[etg] < 0:
                    # mit_rest_config.logger.info(f"Skipped {eachtag} in {sntix} bcoz of {etg}")
                    skipsent = True
                    break
                tag_filters[etg] -= tagsdist[etg]
                # mit_rest_config.logger.info(f"main tag{eachtag} sentence sub {etg} {tagsdist}")
                # mit_rest_config.logger.info(f"{tag_filters}")

            
            if skipsent==True: continue
            already_visited.append(sntix)
            filtered_training.append(training_dataset[sntix])

mit_rest_config.logger.info(f"Length is {len(filtered_training)}, {len(tag_filters.keys())}")
# with open("filtered.json",'w') as fh: json.dump(filtered_training, fh, indent=4)


for i in tqdm(range(0, len(filtered_training))):
    ans = get_entities(filtered_training[i][0], filtered_training[i][1], template_type= mit_rest_config.params["TEMPLATE"])

    tagsdist = counttags(ans[1])
    sent_inclusion_flag = True

    
    data_instances = templatize_function(sentence_text = " ".join(filtered_training[i][0]), entity_text = ans[0], \
                    entity_tags = ans[1], template_type=mit_rest_config.params["TEMPLATE"])
    
    for srcinst, tgtinst, tagname in zip(data_instances[0], data_instances[1], data_instances[2]):
        dataset.append([tagname, srcinst, tgtinst])

mit_rest_config.logger.info(f"Considered {len(filtered_training)} sentences")
mit_rest_config.logger.info(f"Total sents are {len(training_dataset)}")

# In[12]:

import json


# In[13]:


with open(f"{mit_rest_config.params['TRAINING_FILE'].replace('.json','')}_temp{mit_rest_config.params['TEMPLATE']}.json", "w") as fh:
    json.dump(dataset, fh, indent=4)


val_dataset = []
for i in tqdm(range(0, len(dev_dataset))):
    ans = get_entities(dev_dataset[i][0], dev_dataset[i][1], template_type= mit_rest_config.params["TEMPLATE"])

    data_instances = templatize_function(sentence_text = " ".join(dev_dataset[i][0]), entity_text = ans[0], \
                    entity_tags = ans[1], template_type=mit_rest_config.params["TEMPLATE"])
    
    for srcinst, tgtinst, tagname in zip(data_instances[0], data_instances[1], data_instances[2]):
        val_dataset.append([tagname, srcinst, tgtinst])

with open(f"{mit_rest_config.params['DEV_FILE'].replace('.json','')}_temp{mit_rest_config.params['TEMPLATE']}.json", "w") as fh:
    json.dump(val_dataset, fh, indent=4)


test_dataset = []
for i in tqdm(range(0, len(testing_dataset))):
    ans = get_entities(testing_dataset[i][0], testing_dataset[i][1], template_type= mit_rest_config.params["TEMPLATE"])

    data_instances = templatize_function(sentence_text = " ".join(testing_dataset[i][0]), entity_text = ans[0], \
                    entity_tags = ans[1], template_type=mit_rest_config.params["TEMPLATE"])
    
    for srcinst, tgtinst, tagname in zip(data_instances[0], data_instances[1], data_instances[2]):
        test_dataset.append([tagname, srcinst, tgtinst])
    
with open(f"{mit_rest_config.params['TEST_FILE'].replace('.json','')}_temp{mit_rest_config.params['TEMPLATE']}.json", "w") as fh:
    json.dump(test_dataset, fh, indent=4)

with open(f"check.json", "w") as fh:
    json.dump(testing_dataset, fh, indent=4)


# ### Some other references
# 
# 1. https://github.com/Shivanandroy/T5-Finetuning-PyTorch
