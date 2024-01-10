import random
import pandas as pd
object_frases = {}
ds = pd.read_csv("../ppo_tuning/dataset/modified_output_concatenated_long_with_scores.csv", header=0)
df = pd.DataFrame(columns = ['id', 'item_id', 'true_bbox', 'prompt', 'correct', 'rejected', 
                             'iou_correct', 'score_correct', 'harmonic_correct',
                             'iou_rejected', 'score_rejected', 'harmonic_rejected'])
for i in range(len(ds)):
    if ds['item_id'][i] in object_frases.keys():
        object_frases[ds['item_id'][i]].append([i,ds['response'][i]])
    else:
        object_frases[ds['item_id'][i]] = [[i, ds['response'][i]]]
for i in range(len(ds)):
    try:
        prompt = "Send ONLY a single sentence - a rewording of " + ds['response'][i]
        resp1 = ds['response'][i]
        while True:     
            n = random.choice([k for k in range(0,len(object_frases[ds['item_id'][i]]))])
            if object_frases[ds['item_id'][i]][n] != resp1:
                resp2 = object_frases[ds['item_id'][i]][n][1]
                k = object_frases[ds['item_id'][i]][n][0]
                break
        if ds['harmonic_mean'][i] >=  ds['harmonic_mean'][k]:
            df.loc[len(df)] = [i, ds['item_id'][i], ds['true_bbox'][i], prompt, ds['response'][i], ds['response'][k],
                               ds['response_iou'][i], ds['response_score'][i], ds['harmonic_mean'][i],
                              ds['response_iou'][k], ds['response_score'][k], ds['harmonic_mean'][k]]
        else: 
            df.loc[len(df)] = [i, ds['item_id'][i], ds['true_bbox'][i], prompt, ds['response'][k], ds['response'][i],
                               ds['response_iou'][k], ds['response_score'][k], ds['harmonic_mean'][k], 
                               ds['response_iou'][i], ds['response_score'][i], ds['harmonic_mean'][i]]
    except: 
        continue
df.to_csv("./dataset/dpo_final_dataset.csv")