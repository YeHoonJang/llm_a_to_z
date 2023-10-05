# ARC
CUDA_VISIBLE_DEVICES=4 python /home/yehoon/workspace/llm_a_to_z/llama_2/inference.py --inference --model="meta-llama/Llama-2-7b-hf" --dataset="ai2_arc" --dataset_subset="ARC-Easy" --test_split="test" --prompt_style="upstage" --custom_prompt="/home/yehoon/workspace/llm_a_to_z/data/fewshot_role/arc.txt" --output_dir="/home/yehoon/workspace/outputs/" --output_name="0907_ft_llama2_upstage_way" --generated_name="fewshot_role_arc.csv"

# Hellaswag
CUDA_VISIBLE_DEVICES=5 python /home/yehoon/workspace/llm_a_to_z/llama_2/inference.py --inference --model="meta-llama/Llama-2-7b-hf" --dataset="hellaswag" --test_split="validation" --prompt_style="upstage" --custom_prompt="/home/yehoon/workspace/llm_a_to_z/data/fewshot_role/hellaswag.txt" --output_dir="/home/yehoon/workspace/outputs/" --output_name="0907_ft_llama2_upstage_way" --generated_name="fewshot_role_hellaswag.csv"

# MMLU
CUDA_VISIBLE_DEVICES=6 python /home/yehoon/workspace/llm_a_to_z/llama_2/inference.py --inference --model="meta-llama/Llama-2-7b-hf" --dataset="cais/mmlu" --dataset_subset="all" --test_split="test" --prompt_style="upstage" --custom_prompt="/home/yehoon/workspace/llm_a_to_z/data/fewshot_role/mmlu.txt" --output_dir="/home/yehoon/workspace/outputs/" --output_name="0907_ft_llama2_upstage_way" --generated_name="fewshot_role_mmlu.csv"

# Truthful_QA
CUDA_VISIBLE_DEVICES=7 python /home/yehoon/workspace/llm_a_to_z/llama_2/inference.py --inference --model="meta-llama/Llama-2-7b-hf" --dataset="truthful_qa" --dataset_subset="multiple_choice" --test_split="validation" --prompt_style="upstage" --custom_prompt="/home/yehoon/workspace/llm_a_to_z/data/fewshot_role/truthful_qa.txt" --output_dir="/home/yehoon/workspace/outputs/" --output_name="0907_ft_llama2_upstage_way" --generated_name="fewshot_role_truthful_qa.csv"


# Inference once
python /home/yehoon/workspace/llm_a_to_z/llama_2/inference.py --inference --model="meta-llama/Llama-2-13b-hf" --dataset="ai2_arc" --dataset_subset="ARC-Easy" --test_split="test" --prompt_style="upstage" --custom_prompt="../data/fewshot_prompt/arc.txt" --output_dir="/home/yehoon/workspace/outputs/" --output_name="0920_llama_13b" --generated_name="13b_test.csv" --inference_once --prompt="../data/prompt_once.txt"