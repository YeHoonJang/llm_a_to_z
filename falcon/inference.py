# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
# import os
# # Load the fine-tuned model and the tokenizer
# model_path = "/home/yehoon/workspace/llm_a_to_z/falcon/results/falcon-7b-finetuned-4bit"
#
# config = AutoConfig.from_pretrained(os.path.join(model_path, "adapter_config.json"))
# model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
#
# tokenizer = AutoTokenizer.from_pretrained(model_path)
#
# # Define the input text
# input_text = "Hello, how are you?"
#
# # Encode the input text
# input_ids = tokenizer.encode(input_text, return_tensors='pt')
#
# # Get the model's output
# output = model.generate(input_ids, max_length=50, temperature=0.7)
#
# # Decode the output
# decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
#
# print(decoded_output)


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, FalconForCausalLM
import torch
import os

# Specify the fine-tuned model directory
model_path = "/home/yehoon/workspace/llm_a_to_z/falcon/results/falcon-7b-finetuned-4bit"
# config = AutoConfig.from_pretrained(os.path.join(model_path, "adapter_config.json"))
# Load the fine-tuned model and the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = FalconForCausalLM.from_pretrained(model_path)

# Define the input text
input_text = "Hello, how are you?"

# Encode the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)

# Get the model's output
output = model.generate(input_ids, max_length=128, temperature=0.7, do_sample=True, top_k=50, top_p=0.95)

# Decode the output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output)


#
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
# import torch
# import os
#
# # Load the fine-tuned model and the tokenizer
# model_path = "/home/yehoon/workspace/llm_a_to_z/falcon/results/falcon-7b-finetuned-4bit"
#
# config = AutoConfig.from_pretrained(os.path.join(model_path, "adapter_config.json"))
# model = AutoModelForCausalLM.from_config(config)
#
# # Load the model state dict from the "adapter_model.bin" file
# model_state_dict = torch.load(os.path.join(model_path, "adapter_model.bin"))
#
# # Update the model's state dict
# model.load_state_dict(model_state_dict)
#
# tokenizer = AutoTokenizer.from_pretrained(model_path)
#
# # Define the input text
# input_text = "Hello, how are you?"
#
# # Encode the input text
# input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=50)
#
# # Get the model's output
# output = model.generate(input_ids, max_length=50, temperature=0.7)
#
# # Decode the output
# decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
#
# print(decoded_output)
