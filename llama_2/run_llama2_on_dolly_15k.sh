# Train
python /home/yehoon/workspace/llm_a_to_z/llama_2/ft_llama2.py\
  --train\
  --model="meta-llama/Llama-2-7b-hf"\
  --dataset="databricks/databricks-dolly-15k"\
  --output_dir="/home/yehoon/workspace/outputs/"\
  --output_name="0727_llama2_on_dolly_15k"

## Inference
#python /home/yehoon/workspace/llm_a_to_z/llama_2/ft_llama2.py\
#  --inference\
#  --model="meta-llama/Llama-2-7b-hf"\
#  --dataset="databricks/databricks-dolly-15k"\
#  --output_dir="/home/yehoon/workspace/outputs/"\
#  --output_name="0727_llama2_on_dolly_15k"