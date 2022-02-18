import config
import torch
import re
import json

from typing import List, Tuple


class seq2seq_dataset:
    def __init__(self, prl_sents):
        """

        """
        self.prl_sents = prl_sents

    def __len__(self):
        return len(self.prl_sents)
    
    def __getitem__(self, item):
        tagname, src_text, tgt_text = self.prl_sents[item]

        ids = []
        target_tag =[]

        
        model_inputs = config.TOKENIZER(src_text, return_tensors="pt")
        with config.TOKENIZER.as_target_tokenizer():
            labels = config.TOKENIZER(tgt_text, return_tensors="pt").input_ids

        model_inputs["labels"] = labels
        model_inputs["tagnames"] = tagname

        return model_inputs["input_ids"][0], model_inputs["attention_mask"][0], model_inputs["labels"][0], model_inputs["tagnames"]
        # return model_inputs