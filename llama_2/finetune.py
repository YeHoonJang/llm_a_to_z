import os
import argparse
import logging
import torch

from datasets import load_dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer

from utils.config import find_all_linear_names, create_peft_config, create_bnb_config
from utils.utils import print_trainable_parameters, get_max_length
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


def train(opt, model, tokenizer, train_dataset, valid_dataset, output_dir):

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
            logging_steps=opt.logging_steps,
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
            eval_dataset=valid_dataset,
            peft_config=peft_config,
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
            eval_dataset=valid_dataset,
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
    trainer.model.push_to_hub(opt.hub_name)
    trainer.save_model(output_dir)

    # TODO: 여기서 부터 다시!!!
    # model_to_merge = PeftModel.from_pretrained(AutoPeftModelForCausalLM.from_pretrained(opt.model).to(device), output_dir)
    # merged_model = model_to_merge.merge_and_unload()
    # merged_model.save_pretrained(merged_model)


    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action='store_true')
    parser.add_argument("--llama_ft", action='store_true')

    parser.add_argument("--prompt_style", type=str, help="Prompt format style (e.g., dolly, alpaca, upstage ...)")
    parser.add_argument("--custom_prompt", type=str, default="none", help="Path of custom prompt (e.g., prompt.txt)")

    parser.add_argument("--model", type=str, required=True, help="Model Name (e.g., 'meta/llama-2-7b')")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset Name (e.g., 'wikipedia', 'tatsu-lab/alpaca')")
    parser.add_argument("--dataset_subset", type=str, default=None, help="Subset of dataset")
    parser.add_argument("--train_split", type=str, default="train", help="Train split of dataset")
    parser.add_argument("--valid_split", type=str, default="validation", help="Validation split of dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Path where output saved")
    parser.add_argument("--output_name", type=str, required=True, help="Name of the output directory")
    parser.add_argument("--hub_name", type=str, required=True, help="Name of huggingface model where push the local model")
    parser.add_argument("--wandb", type=str, help="Name of the W&B run")


    parser.add_argument("--padding_side", type=str, default=None, choices=["right", "left"], help="Side on which the pad token applied")
    parser.add_argument("--use_flash_attention", action='store_true', help="Using flash attention")

    parser.add_argument("--lora_r", type=int, default=8, help="LoRA R: Dimension of the updated matrices")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA Alpha: Parameter for scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA Dropout: Dropout probability for layers")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max length of sequence")
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch", "no"], help="Checkpoint save strategy")
    parser.add_argument("--epoch", type=int, default=-1, help="Train epoch")
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

    parser.add_argument("--max_memory", type=int, default=10240, help="Maximum of GPU's Memory")
    parser.add_argument("--device_map", type=str, default='auto', help="Device (e.g., 'cpu', 'cuda:1', 'mps', or a GPU ordinal rank like 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    parser.add_argument("--logging_steps", type=int, default=10, help="Number of logging steps")

    opt = parser.parse_args()

    # Download Dataset
    if "dolly" in opt.dataset.lower():
        # dolly dataset has only train split
        train_dataset = load_dataset(opt.dataset, name=opt.dataset_subset, split=f"train[:80%]")
        valid_dataset = load_dataset(opt.dataset, name=opt.dataset_subset, split=f"train[80%:95%]")
    elif "hellaswaq" in opt.dataset.lower():
        # hellaswaq test dataset has no labels
        train_dataset = load_dataset(opt.dataset, name=opt.dataset_subset, split=f"train[:80%]")
        valid_dataset = load_dataset(opt.dataset, name=opt.dataset_subset, split=f"train[80%:]")
    elif "truthful_qa" in opt.dataset.lower():
        # truthful_qa dataset has only validation split
        train_dataset = load_dataset(opt.dataset, name=opt.dataset_subset, split=f"validation[:80%]")
        valid_dataset = load_dataset(opt.dataset, name=opt.dataset_subset, split=f"validation[80%:95%]")
    else:
        train_dataset = load_dataset(opt.dataset, name=opt.dataset_subset, split=f"{opt.train_split}")
        valid_dataset = load_dataset(opt.dataset, name=opt.dataset_subset, split=f"{opt.valid_split}")

    print(f"[Train] Number of prompts: {len(train_dataset)}")
    print(f"[Validation] Number of prompts: {len(valid_dataset)}")

    print(f"[Train] Column names are: {train_dataset.column_names}")
    print(f"[Validation] Column names are: {valid_dataset.column_names}")

    model_name = opt.model

    bnb_config = create_bnb_config()
    model, tokenizer = load_model(opt, model_name, bnb_config)

    # Preprocess dataset
    max_length = get_max_length(model)
    train_dataset = preprocess_dataset(opt, tokenizer, max_length, opt.seed, train_dataset)
    valid_dataset = preprocess_dataset(opt, tokenizer, max_length, opt.seed, valid_dataset)
    output_dir = os.path.join(opt.output_dir, opt.output_name)

    train(opt, model, tokenizer, train_dataset, valid_dataset, output_dir)


if __name__ == "__main__":
    main()
