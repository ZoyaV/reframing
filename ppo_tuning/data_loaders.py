import json
import pandas as pd
from datasets import Dataset


def get_messages():
    with open('../Prompts/reseach_text_patterns/dialogues.json') as f:
        dialogues = json.load(f)
    return dialogues['dialogues']


def iglu2pd(data, inp_column, out_column):
    count_dialogues = len(data)
    pd_format = {inp_column: [], out_column: []}
    for i in range(count_dialogues):
        count_instractions = len(data[i])
        for j in range(count_instractions):
            pd_format[inp_column].append(data[i][j][inp_column])
            pd_format[out_column].append(str(data[i][j][out_column]))
    dataset = pd.DataFrame(pd_format)
    return dataset


class CustomDataset():
    def __init__(self, data, tokenizer, txt_in_len, inp_column, out_column):
        self.tokenizer = tokenizer
        self.txt_in_len = txt_in_len
        self.dataset = Dataset.from_pandas(data)

        for d in self.dataset :
            print(d)
            print()
            print("----------------------")

       # exit()
        self.inp_column = inp_column
        self.out_column = out_column

    def encode_and_decode(self, examples):
        # Encode the reviews
        encoded = self.tokenizer(examples[self.inp_column], truncation=True, padding='max_length', max_length=self.txt_in_len)
        # Add the 'query' column by decoding the 'input_ids'
        encoded['query'] = [self.tokenizer.decode(ids) for ids in encoded['input_ids']]
        return encoded

    def prepare_dataset(self):
        return self.dataset.map(self.encode_and_decode, batched=True)

class SegmentationsDataset(CustomDataset):
    def __init__(self, tokenizer, txt_in_len, inp_column, out_column, prompts):
        data = pd.read_csv('./dataset/prompts.csv')
       # prompt = lambda x: f"""In other words, '{x}' is = """

        def format_with_each_prompt(x):
            return [prompt.format(x) for prompt in prompts]
        data['category_id'] = data['category_id'].map(format_with_each_prompt)
        data = data.explode('category_id')

        super(SegmentationsDataset, self).__init__(data, tokenizer, txt_in_len, inp_column, out_column)


class HFDataset(CustomDataset):
    def __init__(self, tokenizer, txt_in_len, inp_column, out_column):
        data = pd.read_csv('./dataset/hf_prompts.csv')
        data['prompt'] = data['text'].map(lambda x: f"The fixed version of sentence: '{x}' is")
        data = data[data['is_correct'] == 0]
        super(HFDataset, self).__init__(data, tokenizer, txt_in_len, inp_column, out_column)


class IgluDataset(CustomDataset):
    def __init__(self, tokenizer, txt_in_len, inp_column, out_column):
        data = get_messages()
        iglu_data = iglu2pd(data, inp_column, out_column)
        super(IgluDataset, self).__init__(iglu_data, tokenizer, txt_in_len, inp_column, out_column)
