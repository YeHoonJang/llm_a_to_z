import os
import time

from datasets import load_dataset
from smart_open import smart_open
from llmengine import FineTune

import pandas as pd

dataset = load_dataset('derek-thomas/ScienceQA')

# A to Z
choice_prefixes = [chr(ord('A') + i) for i in range(26)]


# (A) pull (B) push ...
def format_options(options, choice_prefixes):
    return ' '.join([f"({c}) {o}" for c, o in zip(choice_prefixes, options)])


def format_prompt(r ,choice_prefixes):
    # r is dictionary of each data
    options = format_options(r["choices"], choice_prefixes)
    return f'''Context: {r["hint"]}\nQuestion: {r["question"]}\nOptions:{options}\nAnswer:'''


def format_response(r, choice_prefixes):
    return choice_prefixes[r["answer"]]


def convert_dataset(ds):
    prompts = [format_prompt(i, choice_prefixes) for i in ds if i['hint'] != '']
    labels = [format_response(i, choice_prefixes) for i in ds if i['hint'] != '']
    df = pd.DataFrame.from_dict({'prompt': prompts, 'response': labels})
    return df


data_path = "/home/yehoon/workspace/data"
df_train = convert_dataset(dataset['train'])
df_valid = convert_dataset(dataset['validation'])

df_train.to_csv(os.path.join(data_path, "scienceqa_train.csv"), index=False)
df_valid.to_csv(os.path.join(data_path, "scienceqa_valid.csv"), index=False)

os.environ["SCALE_API_KEY"] = "clkjnc9gj06pz1aqwav3rhqia"

response = FineTune.create(
    model="llama-2-7b",
    training_file=os.path.join(data_path, "scienceqa_train.csv"),
    validation_file=os.path.join(data_path, "scienceqa_valid.csv"),
    hyperparameters={
        'lr':2e-4,
    },
    suffix="science-qa-llama2"
)
run_id = response.id

while True:
    job_status = FineTune.get(run_id).status
    print(job_status)
    if job_status == "SUCESS":
        break
    time.sleep(30)
