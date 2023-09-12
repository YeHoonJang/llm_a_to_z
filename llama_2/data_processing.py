import pdb
from functools import partial
from transformers import AutoTokenizer

from utils.utils import preprocess_batch


# Pre-processing Dataset
def create_prompt_formats(opt, sample):
    if opt.prompt_style == "upstage":
        INSTRUCTION_KEY = "### User:"
        INPUT_KEY = "### System:"
        RESPONSE_KEY = "### Assistant:"
        END_KEY = ""

    else:
        INSTRUCTION_KEY = "### Instruction:"
        INPUT_KEY = "Input:"
        RESPONSE_KEY = "### Response:"
        END_KEY = "### End"

    # Custom Prompt
    if opt.custom_prompt == "none":
        INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    else:
        with open(opt.custom_prompt, 'r', encoding='utf-8') as f:
            INTRO_BLURB = f.read()

    # Prompt format by dataset
    if "dolly" in opt.dataset.lower():
        blurb = f"{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
        input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
        response = f"{RESPONSE_KEY}\n{sample['response']}"

    elif "scienceqa" in opt.dataset.lower():
        choice_prefixes = [chr(ord('A') + i) for i in range(26)]

        def format_options(options, choice_prefixes):
            return ' '.join([f"({c}) {o}" for c, o in zip(choice_prefixes, options)])

        options = format_options(sample["choices"], choice_prefixes)
        answer = choice_prefixes[sample["answer"]]

        blurb = f"{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY}\n{sample['question']}\n\nOptions:\n{options}"
        input_context = f"{INPUT_KEY}\n{sample['hint']}" if sample["hint"] else None
        response = f"{RESPONSE_KEY}\n{answer}"

    elif "arc" in opt.dataset.lower():
        # options = "\n".join([" ".join([str(label), text]) for label, text in zip(sample["choices"]["label"], sample["choices"]["text"])])
        options = [" ".join([str(label), text]) for label, text in zip(sample["choices"]["label"], sample["choices"]["text"])]

        blurb = f"{INTRO_BLURB} The answer must be one of the one in the list. Answer with the number or symbol of the correct answer without any explanations and quoting mark."
        instruction = f"{INSTRUCTION_KEY}\n{sample['question']}\n\n{str(options)}"
        input_context = ""
        response = f"{RESPONSE_KEY}\n{sample['answerKey']}"

    elif "hellaswag" in opt.dataset.lower():
        # options = "\n".join([" ".join([str(idx), text]) for idx, text in enumerate(sample["endings"])])
        options = [" ".join([str(idx), text]) for idx, text in enumerate(sample["endings"])]

        blurb = f"{INTRO_BLURB} Choose the number to continue the sentence in context and complete it appropriately. The answer must be one of the one in the list. Answer with the number of the correct answer without any explanations."
        instruction = f"{INSTRUCTION_KEY}\n{sample['ctx']}\n\n{str(options)}"
        input_context = ""
        response = f"{RESPONSE_KEY}\n{sample['label']}"

    elif "mmlu" in opt.dataset.lower():
        # options = options = "\n".join([" ".join([str(idx), text]) for idx, text in enumerate(sample["choices"])])
        options = options = [" ".join([str(idx), text]) for idx, text in enumerate(sample["choices"])]

        blurb = f"{INTRO_BLURB} The answer must be one of the one in the list. Answer with the number of the correct answer without any explanations."
        instruction = f"{INSTRUCTION_KEY}\n{sample['question']}\n\n{str(options)}"
        input_context = ""
        response = f"{RESPONSE_KEY}\n{sample['answer']}"

    elif "truthful_qa" in opt.dataset.lower():
        # options = "\n".join([" ".join([str(idx), text]) for idx, text in enumerate(sample["mc1_targets"]["choices"])])
        options = [" ".join([str(idx), text]) for idx, text in enumerate(sample["mc1_targets"]["choices"])]

        blurb = f"{INTRO_BLURB} The answer must be one of the one in the list. Answer with the number or symbol of the correct answer without any explanations."
        instruction = f"{INSTRUCTION_KEY}\n{sample['question']}\n\n{str(options)}"
        input_context = ""
        response = f"{RESPONSE_KEY}\n{sample['mc1_targets']['labels'].index(1)}"

    inference_response = f"{RESPONSE_KEY}\n"
    end = f"{END_KEY}"

    if opt.train:
        parts = [part for part in [blurb, instruction, input_context, response, end] if part]
    elif opt.inference:
        parts = [part for part in [blurb, instruction, input_context, inference_response] if part]

    formatted_prompt = "\n\n".join(parts)

    sample["text"] = formatted_prompt

    return sample

def preprocess_dataset(opt, tokenizer: AutoTokenizer, max_length: int, seed, dataset):

    # Add prompt to each sample
    print("Preprocessing dataset...")

    _create_prompt_formats = partial(create_prompt_formats, opt)

    print(f"Creating prompts for {(opt.dataset).split('/')[-1]} dataset...")
    dataset = dataset.map(_create_prompt_formats)  # , batched=True)
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)

    if opt.train:
        if "dolly" in opt.dataset.lower():
            if opt.llama_ft:
                remove_columns = ["instruction", "context", "response", "category"]
            else:
                remove_columns = ["instruction", "context", "response", "text", "category"]
            dataset = dataset.map(
                _preprocessing_function,
                batched=True,
                remove_columns=remove_columns,
            )
        elif "scienceqa" in opt.dataset.lower():
            if opt.llama_ft:
                remove_columns = ['image', 'question', 'choices', 'answer', 'hint', 'task', 'grade', 'subject', 'topic', 'category', 'skill', 'lecture', 'solution']
            else:
                remove_columns = ["text", 'image', 'question', 'choices', 'answer', 'hint', 'task', 'grade', 'subject', 'topic', 'category', 'skill', 'lecture', 'solution']
            dataset = dataset.map(
                _preprocessing_function,
                batched=True,
                remove_columns=remove_columns
            )

        # Filter out samples that have input_ids exceeding max_length
        dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)


    return dataset