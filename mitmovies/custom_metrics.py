
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def calc_scores(gt_sett_ents, out_sett_ents):
    tot = 0
    matched = []
    for gtent in gt_sett_ents:
        gtents = gtent.strip().split()
        for pred_ent in out_sett_ents:
            predents = pred_ent.strip().split()
            common = intersection(gtents, predents)
            try:
                tot += len(common)/max(len(gtents), len(predents))
                matched.append(common)
            except:
                st.write("Fails at",gt_sett_ents, out_sett_ents)
                st.write(gtents, predents)
    tot/=len(gt_sett_ents)

    return tot, matched

def calc_scores_p_r_full(gt_sett_ents, out_sett_ents):
    tps = [ent for ent in out_sett_ents if ent in gt_sett_ents]
    tp = len(tps)
    gtlen = len(gt_sett_ents)
    predslen = len(out_sett_ents)
    fp = predslen - tp
    precision = tp/predslen
    recall = tp/gtlen

    return precision, recall

def calc_scores_p_r_partial(gt_sett_ents, out_sett_ents):
    score, tps = calc_scores(gt_sett_ents, out_sett_ents) 
    # Look at calc_scores function. It returns the common intersection by splitting the entities
    # into words and checking scores
    score  = score *len(gt_sett_ents)
    tp = score
    gtlen = len(gt_sett_ents)
    predslen = len(out_sett_ents)
    fp = predslen - tp
    precision = tp/predslen
    recall = tp/gtlen

    return precision, recall
