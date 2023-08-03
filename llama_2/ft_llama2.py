import os
import argparse
import logging

import pandas as pd
import torch


from tqdm import tqdm
import bitsandbytes as bnb

from datasets import load_dataset
from functools import partial
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig,\
    DataCollatorForLanguageModeling, GenerationConfig, EarlyStoppingCallback
from trl import SFTTrainer

torch.cuda.empty_cache()

logging.getLogger('codecarbon').setLevel(logging.CRITICAL)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


# Download Model
def load_model(opt, model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f"{opt.max_memory}MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=opt.device_map,
        max_memory={i: max_memory for i in range(n_gpus)}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Needed for Llama tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    if opt.padding_side:
        tokenizer.padding_side = opt.padding_side

    return model, tokenizer


# Pre-processing Dataset
def create_prompt_formats(opt, sample):
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"

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

    inference_response = f"{RESPONSE_KEY}\n"
    end = f"{END_KEY}"

    if opt.train:
        parts = [part for part in [blurb, instruction, input_context, response, end] if part]
    elif opt.inference:
        parts = [part for part in [blurb, instruction, input_context, inference_response] if part]

    formatted_prompt = "\n\n".join(parts)

    sample["text"] = formatted_prompt

    return sample


def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


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


# Create a Bitsandbytes Configuration
def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


def create_peft_config(opt, modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=opt.lora_r,  # dimension of the updated matrices
        lora_alpha=opt.lora_alpha,  # parameter for scaling
        target_modules=modules,
        lora_dropout=opt.lora_dropout,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


# Print Parameter
def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


# Train
def train(opt, model, tokenizer, train_dataset, valid_datset, output_dir):

    use_flash_attention = opt.use_flash_attention

    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(opt, modules)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    args = TrainingArguments(
            per_device_train_batch_size=6 if opt.use_flash_attention else opt.batch_size,
            gradient_accumulation_steps=opt.gradient_step,
            gradient_checkpointing=True,
            warmup_steps=opt.warmup,
            num_train_epochs=opt.epoch if opt.epoch else None,
            max_steps=opt.max_step if opt.max_step else None,
            learning_rate=opt.lr,
            lr_scheduler_type=opt.lr_scheduler_type,
            weight_decay=opt.weight_decay,
            fp16=opt.fp16,
            bf16=opt.bf16,
            tf32=opt.tf32,
            logging_steps=int(len(train_dataset)*0.001),
            output_dir=output_dir,
            optim=opt.optimizer,
            # load_best_model_at_end=True,
            # evaluation_strategy="steps",
            save_strategy=opt.save_strategy,
            # eval_steps=5,
            # save_steps=10,
            run_name=opt.wandb,
    )

    if opt.llama_ft:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=valid_datset,
            max_seq_length=opt.max_seq_length,
            tokenizer=tokenizer,
            # packing=True,
            dataset_text_field="text",
            args=args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

    else:
        # Training parameters
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=valid_datset,
            args=args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
        )

    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

    # Verifying the datatypes before training

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    do_train = True

    # Launch training
    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    trainer.save_model(output_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


def inference(opt, model, tokenizer, test_dataset, output_dir, max_length):
    df = pd.DataFrame(columns=["text"])
    generated = []

    generation_config = GenerationConfig(
        temperature=opt.temperature,
        top_p=opt.top_p,
        top_k=opt.top_k,
    )
    print("Inferencing...")
    if opt.inference_once:
        prompt = opt.prompt
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
            decoded_output = tokenizer.decode(s, skip_special_tokens=True)
            generated.append(decoded_output)

            # print(decoded_output)
        df = pd.concat([df["text"], pd.DataFrame(generated)])
        df.to_csv(os.path.join(opt.output_dir, opt.generated_name), index=False)


def main():
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action='store_true')
    parser.add_argument("--llama_ft", action='store_true')

    parser.add_argument("--inference", action='store_true')

    parser.add_argument("--inference_once", action='store_true')
    parser.add_argument("--prompt", type=str, help="Prompt here, if you want to do inferencing once")


    parser.add_argument("--model", type=str, required=True, help="Model Name (e.g., 'meta/llama-2-7b')")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset Name (e.g., 'wikipedia', 'tatsu-lab/alpaca')")
    parser.add_argument("--output_dir", type=str, required=True, help="Path where output saved")
    parser.add_argument("--output_name", type=str, required=True, help="Name of the output directory")
    parser.add_argument("--generated_name", type=str, help="Name of the generated output directory")
    parser.add_argument("--wandb", type=str, help="Name of the W&B run")

    parser.add_argument("--padding_side", type=str, default=None, choices=["right", "left"], help="Side on which the pad token applied")
    parser.add_argument("--use_flash_attention", action='store_true', help="Using flash attention")

    parser.add_argument("--lora_r", type=int, default=8, help="LoRA R: Dimension of the updated matrices")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA Alpha: Parameter for scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA Dropout: Dropout probability for layers")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max length of sequence")
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch", "no"], help="Checkpoint save strategy")
    parser.add_argument("--epoch", type=int, default=3, help="Train epoch")
    parser.add_argument("--max_step", type=int, default=-1, help="Max step")
    parser.add_argument("--gradient_step", type=int, default=4, help="Gradient accumulation step")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup step")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="Type of learning rate schedule")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay ratio")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit", help="Optimizer")
    parser.add_argument("--fp16", action='store_true', help="Using fp16")
    parser.add_argument("--bf16", action='store_true', help="Using bf16")
    parser.add_argument("--tf32", action='store_true', help="Using tf32")

    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature range 0.0~1.0")
    parser.add_argument("--top_k", type=int, default=50, help="Top kth words")
    parser.add_argument("--top_p", type=int, default=0.95, help="Top probability words")

    parser.add_argument("--max_memory", type=int, default=10240, help="Maximum of GPU's Memory")
    parser.add_argument("--device_map", type=str, default='auto',
                        help="Device (e.g., 'cpu', 'cuda:1', 'mps', or a GPU ordinal rank like 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")

    opt = parser.parse_args()

    # Download Dataset
    train_dataset = load_dataset(opt.dataset, split="train[:80%]")
    valid_datset = load_dataset(opt.dataset, split="train[80%:95%]")
    test_dataset = load_dataset(opt.dataset, split="train[95%:]")

    if opt.train:
        print(f"[Train] Number of prompts: {len(train_dataset)}")
        print(f"[Validation] Number of prompts: {len(valid_datset)}")

        print(f"[Train] Column names are: {train_dataset.column_names}")
        print(f"[Validation] Column names are: {valid_datset.column_names}")

        model_name = opt.model

        bnb_config = create_bnb_config()
        model, tokenizer = load_model(opt, model_name, bnb_config)

        # Preprocess dataset
        max_length = get_max_length(model)
        train_dataset = preprocess_dataset(opt, tokenizer, max_length, opt.seed, train_dataset)
        valid_dataset = preprocess_dataset(opt, tokenizer, max_length, opt.seed, valid_datset)
        output_dir = os.path.join(opt.output_dir, opt.output_name)

        train(opt, model, tokenizer, train_dataset, valid_dataset, output_dir)

    elif opt.inference:
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
        model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(opt.model)
        tokenizer.pad_token = tokenizer.eos_token

        max_length = get_max_length(model)
        test_dataset = preprocess_dataset(opt, tokenizer, max_length, opt.seed, test_dataset)
        inference(opt, model, tokenizer, test_dataset, output_dir, max_length)


if __name__ == "__main__":
    main()
