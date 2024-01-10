import pandas as pd
objects = []
original =pd.read_csv("../ppo_tuning/dataset/gold/text.tsv", sep='\t', header=0)
original = original.drop(['hit_id', 'worker_id', 'worktime_s'], axis =1)
df1 =pd.read_csv("../ppo_tuning/dataset/modified_output.csv", header=0)
df2 =pd.read_csv("../ppo_tuning/dataset/modified_output2.csv", header=0)
df1 = df1.drop('item_id', axis=1)
result = pd.concat([df1, df2], ignore_index=True, copy=False)
for i in range(len(result)):
    for j in range(len(original)):
        if result['preferenced'][i] == original['text'][j]:
            objects.append(original['item_id'][j])
            break
result['item_id'] = objects
result.to_csv("../ppo_tuning/dataset/modified_output_concatenated.csv") 