import pandas as pd
import datasets
import argparse
import os

from utils.metrics import get_accuracy, get_chatgpt_accuracy


# ARC
def arc_evaluate(opt):
    arc_df = pd.read_csv(os.path.join("../../outputs/", opt.file_name))
    arc_answer = datasets.load_dataset("ai2_arc", name="ARC-Easy", split="test")
    arc_df["response"] = arc_df["0"].apply(lambda x: x.split("### Assistant:\n")[-1])
    answer = arc_answer["answerKey"]

    if opt.first_char_acc:
        accuracy = get_accuracy(arc_df, answer)
        print(f"Accuracy: {accuracy}")
    elif opt.chatgpt_acc:
        chatgpt_acc = get_chatgpt_accuracy(opt, arc_df, answer)
        print(f"Accuracy: {chatgpt_acc}")


# HellaSwag
def hellaswag_evaluate(opt):
    hella_df = pd.read_csv(os.path.join("../../outputs/", opt.file_name))
    hella_answer = datasets.load_dataset("hellaswag", split="validation")
    hella_df["response"] = hella_df["0"].apply(lambda x: x.split("### Assistant:\n")[-1])
    answer = hella_answer["label"]

    if opt.first_char_acc:
        accuracy = get_accuracy(hella_df, answer)
        print(f"Accuracy: {accuracy}")
    elif opt.chatgpt_acc:
        chatgpt_acc = get_chatgpt_accuracy(opt, hella_df, answer)
        print(f"Accuracy: {chatgpt_acc}")


# MMLU
def mmlu_evaluate(opt):
    mmlu_df = pd.read_csv(os.path.join("../../outputs/", opt.file_name))
    mmlu_answer = datasets.load_dataset("cais/mmlu", name="all", split="test")
    mmlu_df["response"] = mmlu_df["0"].apply(lambda x: x.split("### Assistant:\n")[-1])
    answer = mmlu_answer["answer"]

    if opt.first_char_acc:
        accuracy = get_accuracy(mmlu_df, answer)
        print(f"Accuracy: {accuracy}")
    elif opt.chatgpt_acc:
        chatgpt_acc = get_chatgpt_accuracy(opt, mmlu_df, answer)
        print(f"Accuracy: {chatgpt_acc}")



# TruthfulQA
def truthfulqa_evaluate(opt):
    truthfulqa_df = pd.read_csv(os.path.join("../../outputs/", opt.file_name))
    truthfulqa_answer = datasets.load_dataset("truthful_qa", name="multiple_choice", split="validation[:95%]")
    truthfulqa_df["response"] = truthfulqa_df["0"].apply(lambda x: x.split("### Assistant:\n")[-1])
    answer = [str(i["labels"].index(1)) for i in truthfulqa_answer["mc1_targets"]]

    if opt.first_char_acc:
        accuracy = get_accuracy(truthfulqa_df, answer)
        print(f"Accuracy: {accuracy}")
    elif opt.chatgpt_acc:
        chatgpt_acc = get_chatgpt_accuracy(opt, truthfulqa_df, answer)
        print(f"Accuracy: {chatgpt_acc}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--arc", action='store_true')
    parser.add_argument("--hellaswag", action='store_true')
    parser.add_argument("--mmlu", action='store_true')
    parser.add_argument("--truthfulqa", action='store_true')

    parser.add_argument("--first_char_acc", action='store_true')
    parser.add_argument("--chatgpt_acc", action='store_true')

    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--acc_file_path", type=str, required=True)
    parser.add_argument("--acc_file_name", type=str, required=True)

    opt = parser.parse_args()

    if opt.arc:
        arc_evaluate(opt)
    elif opt.hellaswag:
        hellaswag_evaluate(opt)
    elif opt.mmlu:
        mmlu_evaluate(opt)
    elif opt.truthfulqa:
        truthfulqa_evaluate(opt)



if __name__ == "__main__":
    main()
