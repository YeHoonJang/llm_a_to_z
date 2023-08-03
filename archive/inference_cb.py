from peft import LoraConfig
import torch
import pandas as pd


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig

# Specify the fine-tuned model directory
model_path = "/home/yehoon/workspace/llm_a_to_z/llama_2/results_llama/llama2-7b-finetuned-4bit_cb"
model_name = "meta-llama/Llama-2-7b-hf"

# Load the fine-tuned model and the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
)

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64


peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

model = LlamaForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    load_in_8bit=True,
)

infer_test = pd.read_csv("/home/yehoon/workspace/data/infer_test.csv")
# Define the input text
for _, row in infer_test[:-1].iterrows():

    input_text = row.text

    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=1000).to("cuda")

    # Get the model's output
    output = model.generate(input_ids, max_length=1024, temperature=0.5, do_sample=True, top_k=50, top_p=0.95)

    # Decode the output
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    print(decoded_output)
    print("-----------------------------")
