import torch
import copy
import config
from tqdm import tqdm
from typing import List, final
import custom_metrics

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():  #BioBERT is taking alot of space
            if k == "tagnames": continue
            data[k] = v.to(device)
        optimizer.zero_grad()
        # _, loss = model(data["ids"],data["mask"], data["token_type_ids"], data["target_tag"])
        outs = model(input_ids = data["input_ids"],attention_mask = data["attention_mask"],labels=data["labels"]) #forward pass
        loss = outs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items(): #BioBERT is taking alot of space
            if k == "tagnames": continue
            data[k] = v.to(device)
        outs = model(input_ids = data["input_ids"],attention_mask = data["attention_mask"],labels=data["labels"]) #forward pass
        loss = outs.loss
        final_loss += loss.item()
    return final_loss / len(data_loader)

def calc_scores(data_loader, model, device):

    scores_fullmatch = dict()
    scores_partialmatch = dict()

    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items(): #BioBERT is taking alot of space
            if k == "tagnames": continue
            data[k] = v.to(device)

        outs = model.generate(input_ids = data["input_ids"],attention_mask = data["attention_mask"]) #forward pass

        for out, tag, gt in zip(outs, data["tagnames"], data["labels"]):
            gt = gt.cpu().tolist()
            gt = [num for num in gt if num!=-100]
            pred = config.TOKENIZER.decode(out, skip_special_tokens=True)
            gt = config.TOKENIZER.decode(gt, skip_special_tokens=True)

            custom_metrics.calc_scores(gt.strip().split(",") ,pred.strip().split(","))
            full_p, full_r = custom_metrics.calc_scores_p_r_full(gt.strip().split(",") ,pred.strip().split(","))
            pr_p, pr_r = custom_metrics.calc_scores_p_r_partial(gt.strip().split(",") ,pred.strip().split(","))

            scores_fullmatch[tag] = scores_fullmatch.get(tag, []) + [[full_p, full_r]]
            scores_partialmatch[tag] = scores_partialmatch.get(tag, []) + [[pr_p, pr_r]]

    finalsummary = dict()
    for tag in scores_fullmatch:
        scores = scores_fullmatch[tag]
        overall_p = sum([p for p,r in scores])/len(scores)
        overall_r = sum([r for p,r in scores])/len(scores)
        fullscore = (f"Full-String-Match \nTag {tag} ==> Precision {overall_p} && Recall {overall_r}")
        config.logger.info(fullscore)

        scores = scores_partialmatch[tag]
        overall_p_pr = sum([p for p,r in scores])/len(scores)
        overall_r_pr = sum([r for p,r in scores])/len(scores)
        partscore = (f"Partial-String-Match \nTag {tag} ==> Precision {overall_p} && Recall {overall_r}")
        config.logger.info(partscore)

        full_f1 = 2*(overall_p * overall_r)/(overall_r+overall_p+1e-8)
        partial_f1 = 2*(overall_p_pr*overall_r_pr)/(overall_r_pr+overall_p_pr+1e-8)

        finalsummary[tag] = {"full_f1": full_f1, "partial_f1": partial_f1,\
            "full_overall_p": overall_p, "full_overall_r": overall_r,\
            "partial_overall_p": overall_p_pr, "partial_overall_r": overall_r_pr}

    return finalsummary


def calc_scores_wo_none(data_loader, model, device):

    scores_fullmatch = dict()
    scores_partialmatch = dict()

    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items(): #BioBERT is taking alot of space
            if k == "tagnames": continue
            data[k] = v.to(device)

        outs = model.generate(input_ids = data["input_ids"],attention_mask = data["attention_mask"]) #forward pass

        for out, tag, gt in zip(outs, data["tagnames"], data["labels"]):
            gt = gt.cpu().tolist()
            gt = [num for num in gt if num!=-100]
            pred = config.TOKENIZER.decode(out, skip_special_tokens=True)
            gt = config.TOKENIZER.decode(gt, skip_special_tokens=True)
            
            if gt == "None": continue

            custom_metrics.calc_scores(gt.strip().split(",") ,pred.strip().split(","))
            full_p, full_r = custom_metrics.calc_scores_p_r_full(gt.strip().split(",") ,pred.strip().split(","))
            pr_p, pr_r = custom_metrics.calc_scores_p_r_partial(gt.strip().split(",") ,pred.strip().split(","))

            scores_fullmatch[tag] = scores_fullmatch.get(tag, []) + [[full_p, full_r]]
            scores_partialmatch[tag] = scores_partialmatch.get(tag, []) + [[pr_p, pr_r]]

    finalsummary = dict()
    for tag in scores_fullmatch:
        scores = scores_fullmatch[tag]
        overall_p = sum([p for p,r in scores])/len(scores)
        overall_r = sum([r for p,r in scores])/len(scores)
        fullscore = (f"Full-String-Match \nTag {tag} ==> Precision {overall_p} && Recall {overall_r}")
        # config.logger.info(fullscore)

        scores = scores_partialmatch[tag]
        overall_p_pr = sum([p for p,r in scores])/len(scores)
        overall_r_pr = sum([r for p,r in scores])/len(scores)
        partscore = (f"Partial-String-Match \nTag {tag} ==> Precision {overall_p} && Recall {overall_r}")
        # config.logger.info(partscore)

        full_f1 = 2*(overall_p * overall_r)/(overall_r+overall_p+1e-8)
        partial_f1 = 2*(overall_p_pr*overall_r_pr)/(overall_r_pr+overall_p_pr+1e-8)

        finalsummary[tag] = {"full_f1": full_f1,\
            "full_overall_p": overall_p, "full_overall_r": overall_r,\
            "partial_overall_p": overall_p_pr, "partial_overall_r": overall_r_pr,\
            "partial_f1": partial_f1}

    config.logger.info("======================================")
    for tg in finalsummary:
        config.logger.info(f"{tg} => <3{finalsummary[tg]}")
    config.logger.info("======================================")

    return finalsummary

def return_predictions(data_loader, model, device):
    """We collate all the preds of diff tag templates
    and merge preds and send them

    Args:
        data_loader ([DataLoader]): [This is only for templates of one example]
        model ([type]): [model]
        device ([type]): [device: cuda or not]

    Returns:
        [type]: [description]
    """
    model.eval()

    for data in data_loader:
        for k, v in data.items(): #BioBERT is taking alot of space
            if k == "tagnames": continue
            data[k] = v.to(device)

        outs = model.generate(input_ids = data["input_ids"],attention_mask = data["attention_mask"]) #forward pass

        preds = []
        for out, tag, gt in zip(outs, data["tagnames"], data["labels"]):
            gt = gt.cpu().tolist()
            gt = [num for num in gt if num!=-100]
            pred = config.TOKENIZER.decode(out, skip_special_tokens=True)
            gt = config.TOKENIZER.decode(gt, skip_special_tokens=True)
            if pred == 'None' or pred=="": continue
            pred = pred.split(",")
            preds+= [(eachpred, tag) for eachpred in pred]

    return preds

def return_predictions_overall(data_loader, model, device, numtags):
    """We collate all the preds of diff tag templates
    and merge preds and send them
    We divide by num tags to collect tags per sample
    Args:
        data_loader ([DataLoader]): [This is only for templates of one example]
        model ([type]): [model]
        device ([type]): [device: cuda or not]

    Returns:
        [type]: [description]
    """
    model.eval()

    preds = []
    for data in data_loader:
        for k, v in data.items(): #BioBERT is taking alot of space
            if k == "tagnames": continue
            data[k] = v.to(device)

        outs = model.generate(input_ids = data["input_ids"],attention_mask = data["attention_mask"]) #forward pass

        for out, tag, gt in zip(outs, data["tagnames"], data["labels"]):
            gt = gt.cpu().tolist()
            gt = [num for num in gt if num!=-100]
            pred = config.TOKENIZER.decode(out, skip_special_tokens=True)
            if pred == "None" or pred=="":
                preds.append("")
            else:
                pred = pred.split(",")
                preds.append([(eachpred, tag) for eachpred in pred])

    preds = [preds[i:i+numtags] for i in range(0,len(preds),numtags)]
    assert len(preds[0])==numtags
    no_none_preds = []
    for sentents in preds: # List of lists(of len-numtags)
        persent = []
        for sentent in sentents: # sentents is a list of ents per sent
            if sentent=="": continue
            persent += sentent
        
        no_none_preds.append(persent)

    return no_none_preds
