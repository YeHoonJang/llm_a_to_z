import glob
import os

import pandas as pd
from datasets import load_dataset

data_path = "/home/yehoon/workspace/llm_a_to_z/llama_2/total.csv"

df = pd.read_csv(data_path)
df = df.fillna("")
text_col = []
for _, row in df.iterrows():
    # TODO: 모든 데이터 key 바꾸기!
    prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    instruction = str(row["instruction"])
    input_query = str(row["inputs"])
    response = str(row["response"])

    if len(input_query.strip()) == 0:
        text = f"{prompt}\n### Instruction:\n{instruction}\n### Response:\n{response}"
    else:
        text = f"{prompt}\n### Instruction:\n{instruction}\n### Input:\n{input_query}\n### Response:\n{response}"

    text_col.append(text)
df.loc[:, "text"] = text_col

df.to_csv("/home/yehoon/workspace/llm_a_to_z/llama_2/train.csv", index=False)