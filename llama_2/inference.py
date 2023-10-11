import os
import argparse
import logging
import torch

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

from utils.utils import get_max_length
from data_processing import preprocess_dataset


# empty cache
torch.cuda.empty_cache()

# remove codecarbon log
logging.getLogger('codecarbon').setLevel(logging.CRITICAL)

# GPU check
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def inference(opt, model, tokenizer, test_dataset):
    df = pd.DataFrame(columns=["text"])
    generated = []

    generation_config = GenerationConfig(
        temperature=opt.temperature,
        top_p=opt.top_p,
        top_k=opt.top_k,
    )
    print("Inference...")
    if opt.inference_once:
        with open(opt.prompt, 'r', encoding='utf-8') as f:
            prompt = f.read()
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            max_new_tokens=256
        )
        s = output.sequences[0]
        decoded_output = tokenizer.decode(s, skip_special_tokens=True)
        print(decoded_output)
    else:
        for i in tqdm(test_dataset["text"]):
            inputs = tokenizer(i, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)

            output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                max_new_tokens=256
            )
            s = output.sequences[0]
            # print(output)
            decoded_output = tokenizer.decode(s, skip_special_tokens=True)
            generated.append(decoded_output)

            # print(decoded_output)
        df = pd.concat([df["text"], pd.DataFrame(generated)])
        df.to_csv(os.path.join(opt.output_dir, opt.generated_name), index=False)

    # TODO: evaluate part begin
    # if opt.evaluate:
    #    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action='store_true')

    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--inference_once", action='store_true')
    parser.add_argument("--prompt", type=str, help="Path of prompt here, if you want to do inferencing once")

    parser.add_argument("--evaluate", action='store_true')

    parser.add_argument("--prompt_style", type=str, help="Prompt format style (e.g., dolly, alpaca, upstage ...)")
    parser.add_argument("--custom_prompt", type=str, default="none", help="Path of custom prompt (e.g., prompt.txt)")
    parser.add_argument("--intro_blurb", type=str, default="none", help="Path of custom prompt (e.g., prompt.txt)")

    parser.add_argument("--model", type=str, required=True, help="Model Name (e.g., 'meta/llama-2-7b')")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset Name (e.g., 'wikipedia', 'tatsu-lab/alpaca')")
    parser.add_argument("--dataset_subset", type=str, default=None, help="Subset of dataset")
    parser.add_argument("--test_split", type=str, default="test", help="Split of dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Path where output saved")
    parser.add_argument("--output_name", type=str, required=True, help="Name of the output directory")
    parser.add_argument("--generated_name", type=str, help="Name of the generated output directory")

    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature range 0.0~1.0")
    parser.add_argument("--top_k", type=int, default=50, help="Top kth words")
    parser.add_argument("--top_p", type=int, default=0.95, help="Top probability words")
    parser.add_argument("--max_length", type=int, default=None, help="Max length of prompt")

    parser.add_argument("--device_map", type=str, default='auto',
                        help="Device (e.g., 'cpu', 'cuda:1', 'mps', or a GPU ordinal rank like 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")

    opt = parser.parse_args()

    if opt.custom_prompt:
        with open(opt.custom_prompt, 'r', encoding='utf-8') as f:
            opt.intro_blurb = f.read()

    # Download Dataset
    if "dolly" in opt.dataset.lower():
        # dolly dataset has only train splits
        test_dataset = load_dataset(opt.dataset, name=opt.dataset_subset, split=f"train[:95%]")
    elif "hellaswag" in opt.dataset.lower():
        # hellaswaq test dataset has no labels
        test_dataset = load_dataset(opt.dataset, name=opt.dataset_subset, split=f"validation")
    elif "truthful_qa" in opt.dataset.lower():
        # truthful_qa dataset has only validation splits
        # test_dataset = load_dataset(opt.dataset, name=opt.dataset_subset, split=f"validation[95%:]")
        test_dataset = load_dataset(opt.dataset, name=opt.dataset_subset, split=f"validation[:95%]")
    else:
        test_dataset = load_dataset(opt.dataset, name=opt.dataset_subset, split=f"{opt.test_split}")

    print(f"[Test] Number of prompts: {len(test_dataset)}")
    print(f"[Test] Column names are: {test_dataset.column_names}")
    model_name = opt.model
    output_dir = os.path.join(opt.output_dir, opt.output_name)

    lora_weights = output_dir

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=False,
        torch_dtype=torch.float16,
        device_map=opt.device_map,
    )
    if "chat" not in model_name.lower():
        print("Peft setting ... ")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    else:
        print(f"Using original chat model: {model_name}")

    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(opt.model)
    tokenizer.pad_token = tokenizer.eos_token

    if opt.max_length is None:
        max_length = get_max_length(model)
    else:
        max_length = opt.max_length
    test_dataset = preprocess_dataset(opt, tokenizer, max_length, opt.seed, test_dataset)
    inference(opt, model, tokenizer, test_dataset)

    # TODO: evaluate part begin


if __name__ == "__main__":
    main()
