import random
import time
import os
from tqdm import tqdm

import openai
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv('env/settings.env')
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_accuracy(model_df, answer):
    response = list(model_df["response"].str[0])
    acc_df = pd.DataFrame({"answer": answer, "response": response})
    acc_df["accuracy"] = (acc_df['answer'] == acc_df['response']).astype(int)

    return round(acc_df["accuracy"].mean() * 100, 2)


def get_chatgpt_accuracy(opt, model_df, answer_column):
    acc_list = []
    sample_indices = random.sample(range(len(model_df)), int(len(model_df) * 0.5))

    chunk_size = len(sample_indices) // int(len(sample_indices)*0.1)

    for i in tqdm(range(0, len(sample_indices), chunk_size)):
        chunk_indices = sample_indices[i:i + chunk_size]

        for idx in tqdm(chunk_indices, leave=False):
            model_answer = "### User:\n" + (model_df.loc[idx, "0"].split("### User:\n")[-1])
            answer = answer_column[idx]

            prompt = f"""You are a helpful AI grader.\n\nThe <Model> part below is the result of the model taking input and answering it. The true answer is given as <Answer>. Your job is to score whether the '### Assistant' within <Model> gives the correct answer to question of '### User' and the choices below. You should say only 1 for True and 0 for False without any additional explanation. If you're not sure if what the '### Assistant' said is correct or not, choose 0.\n\n<Answer>{answer}</Answer>\n\n<Model>\n{model_answer}\n</Model>"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
                max_tokens=2048,
                top_p=1,
            )

            acc_list.append(int(response["choices"][0]["message"]["content"]))
            time.sleep(1)
        with open(os.path.join(opt.acc_file_path, opt.acc_file_name), "w") as file:
            for item in acc_list:
                file.write(str(item) + "\n")
        time.sleep(5)

    return round(sum(acc_list)/len(acc_list)*100, 2)
