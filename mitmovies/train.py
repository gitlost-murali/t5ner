
from cgi import test
import config
import custom_tools
import mitmovie_make_t5_template as make_t5_template
config.logger.info("Experiment setup config loaded...")
config.logger.info("Tokenizer loaded...")
import dataset
config.logger.info("Dataset support loaded...")
import engine
config.logger.info("Training fns loaded...")

config.logger.info("All supporting co-modules loaded...")

import torch
import torch.nn as nn
import json
import re
import numpy as np

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from transformers import AutoModelWithLMHead, T5ForConditionalGeneration
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = config.params["CUDA_VISIBLE_DEVICES"]

from random import shuffle
from pathlib import Path

config.logger.info("Importing done!!")

from datetime import datetime
from tqdm import tqdm

now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M")

exp_folder = f"../moviemit-nontriv_{make_t5_template.filter_limit}_runs/{config.params['TEMPLATE']}/{date_time}"
Path(exp_folder).mkdir(parents=True, exist_ok=True)

import wandb

wandb.init(project=f"moviemit-nontriv-res-size-{make_t5_template.filter_limit}", entity="manoharupv")

wandb.config = config.params

def find_tags(sentence):
  return True if len(re.findall("<[aA-zZ]{1,}>.*</[aA-zZ]{1,}>",sentence)) else False

def cleanse_url_tags_train(filepath: str) -> List[Tuple[str, str]]:
    """From the JSON file containing src,tgt parallel sentences,
        remove any pair that contains url in src/tgt lang sentence.

    Args:
        filepath (str): Location of the JSON file

    Returns:
        List[Tuple(str, str)]: List of Tuple of src, tgt sentence 
    """
    config.logger.info("Cleaning...")
    with open(filepath,"r") as fh: 
        parallel_sents = json.load(fh)

    updated_parallel_sents = []
    for tgname, src, tgt in parallel_sents:
        updated_parallel_sents.append((tgname, src,tgt))

    return updated_parallel_sents


from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
    (xx, yy,zz, ii) = zip(*batch)
    # xx is input_ids
    # yy is attention mask
    # zz is labels
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    z_lens = [len(z) for z in zz]

    # config.logger.info(xx,yy,zz)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=config.TOKENIZER.pad_token_id)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=config.TOKENIZER.pad_token_id)
    zz_pad = pad_sequence(zz, batch_first=True, padding_value=-100)

    # replace padding token id's of the labels by -100

    return {"input_ids":xx_pad, "attention_mask": yy_pad, "labels": zz_pad, "tagnames":ii}

def calculate_tagwise_data_old(instancess):
    tagdict = dict()
    for _, eachinstance_tags in instancess:
        for tag in eachinstance_tags:
            tagdict[tag] = tagdict.get(tag,0) + 1
    
    return tagdict

def calculate_tagwise_data(instancess):
    tagdict = dict()
    for tagname, _, eachinstance_entities in instancess:
        if eachinstance_entities != "None":
            tagdict[tagname] = tagdict.get(tagname, 0) + len(eachinstance_entities.split(","))

    return tagdict


def whole_train(size, numtags, train_sents, val_sents, test_sents):

    train_sents = train_sents[:size]
    tags_count = calculate_tagwise_data(train_sents)

    config.logger.info("==========================================")
    config.logger.info(f"Tag distribution to the input of the iteration with {size/numtags} instances is ")
    config.logger.info(tags_count)
    config.logger.info("==========================================")

    config.logger.info("Creating Datasets & Dataloaders ...")
    train_dataset = dataset.seq2seq_dataset(train_sents)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.params["TRAIN_BATCH_SIZE"], num_workers=1,
        collate_fn=pad_collate
    )

    valid_dataset = dataset.seq2seq_dataset(val_sents)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.params["VALID_BATCH_SIZE"], num_workers=1,
        collate_fn=pad_collate

    )

    test_dataset = dataset.seq2seq_dataset(test_sents)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.params["VALID_BATCH_SIZE"], num_workers=1,
        collate_fn=pad_collate

    )

    config.logger.info("Data Loaders initialized..")
    config.logger.info("Loading Model...")
    model = T5ForConditionalGeneration.from_pretrained(config.params["BASE_MODEL_PATH"])
    device = torch.device("cuda" if config.params["CUDA"] else "cpu")

    model.to(device) #Model is taking alot of space, so go for Half-Precision if possible
    if config.params["HALF_PRECISION"]:
        model = model.half() 
        #https://github.com/pytorch/pytorch/issues/25946#issuecomment-530227975


    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_sents) / config.params["TRAIN_BATCH_SIZE"] * config.params["EPOCHS"])
    optimizer = AdamW(optimizer_parameters, lr=config.params["lr"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_modelname = f'{exp_folder}/{Path(config.params["BASE_MODEL_PATH"]).stem}_finetuned_tagcnt_{make_t5_template.filter_limit}'
    config.TOKENIZER.save_pretrained(best_modelname)

    config.logger.info("Training..")

    snooze_switch = 0
    snooze_limit = 5

    def test_standard_speedup():
        "Testing using standard metric..."
        test_sents_st = make_t5_template.testing_dataset[:int(len(test_sents)/numtags)]
        test_st_forsent = []
        standard_sent_pred_gt = []
        for eachsentence, eachsentencetags in tqdm(test_sents_st):
            ans = make_t5_template.get_entities(eachsentence, eachsentencetags, template_type= config.params["TEMPLATE"])

            data_instances = make_t5_template.templatize_function(sentence_text = " ".join(eachsentence), entity_text = ans[0], \
                            entity_tags = ans[1], template_type=config.params["TEMPLATE"])

            test_st_forsent += list(zip(data_instances[2], data_instances[0], data_instances[1]))

        test_dataset_st = dataset.seq2seq_dataset(test_st_forsent)
        test_data_loader_st = torch.utils.data.DataLoader(
            test_dataset_st, batch_size=config.params["VALID_BATCH_SIZE"], num_workers=1,
            collate_fn=pad_collate, shuffle=False
        )
        
        predsphrases_withtags = engine.return_predictions_overall(test_data_loader_st, model, device, numtags)
        assert len(predsphrases_withtags) == len(test_sents_st)
        for sentenceinstance, predsphrases_withtags_persent in zip(test_sents_st, predsphrases_withtags):
            eachsentence, eachsentencetags = sentenceinstance
            predtags_iob = custom_tools.find_tags_andiob(predsphrases_withtags_persent, eachsentence)
            standard_sent_pred_gt.append([eachsentence, predtags_iob, eachsentencetags, predsphrases_withtags_persent])

        with open("check-preds-gt-speedup.json","w") as fh: json.dump(standard_sent_pred_gt, fh,indent=4)
        y_pred = [instancetags for _,instancetags,_,_ in standard_sent_pred_gt]
        y_true = [instancetags for _, _, instancetags,_ in standard_sent_pred_gt]

        fscore, cm = custom_tools.calculate_seqeval(y_true, y_pred)
        config.logger.info("============================================")
        config.logger.info("F-score using standard `seqeval` ")
        config.logger.info("============================================")
        config.logger.info(fscore)
        config.logger.info(cm)

        return fscore, cm

    best_loss = np.inf
    for epoch in range(config.params["EPOCHS"]):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        valid_loss = engine.eval_fn(valid_data_loader, model, device)
        losses_tv = f"{train_loss, valid_loss}"
        config.logger.info(losses_tv)
        wandb.log({"train loss": train_loss, "val loss": valid_loss})

        snooze_switch+=1

        if valid_loss < best_loss:
            snooze_switch = 0
            best_loss = valid_loss
            model.save_pretrained( best_modelname )

        if snooze_switch>= snooze_limit:
            config.logger.info("============================================")
            config.logger.info("Training stopped due to increasing val-loss")
            config.logger.info("============================================")
            break

    # finalsummary = engine.calc_scores(test_data_loader, model, device)
    config.logger.info("============================================")
    config.logger.info("Scores without None")
    config.logger.info("============================================")
    finalsummary = engine.calc_scores_wo_none(test_data_loader, model, device)

    fscore, cm = test_standard_speedup()
    wandb.run.summary["test_scores"] = cm
    wandb.run.summary["fscore"] = fscore

    config.logger.info("="*20)
    shuffle(val_sents)
    valcount = 0
    for valsent in val_sents:
        if valsent[-1] == "None": continue
        if valcount == 20: break
        valcount += 1
        inputs = config.TOKENIZER.encode(valsent[1], return_tensors="pt")
        outputs = model.generate(inputs.cuda())
        config.logger.info(valsent)
        output = "Output ->",config.TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        config.logger.info(output)
        config.logger.info("="*20)

    with open(exp_folder+"/"+config.logger_file,"w") as fh:
        with open("logs/"+config.logger_file,"r") as fl:
            prog = fl.read()
        fh.write(prog)

if __name__ == "__main__":
    numtags = len(make_t5_template.tag_descriptor.keys())

    train_sents = cleanse_url_tags_train(filepath=f'{config.params["TRAINING_FILE"].replace(".json","")}_temp{config.params["TEMPLATE"]}.json')

    val_sents = cleanse_url_tags_train(filepath=f'{config.params["DEV_FILE"].replace(".json","")}_temp{config.params["TEMPLATE"]}.json')
    test_sents = cleanse_url_tags_train(filepath=f'{config.params["TEST_FILE"].replace(".json","")}_temp{config.params["TEMPLATE"]}.json')

    replications = 1
    if config.params["TEMPLATE"] == 2:
        replications = numtags

    size = 9999999 #MAX size
    size = size * replications
    whole_train(size, numtags, train_sents, val_sents, test_sents)