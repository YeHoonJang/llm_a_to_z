python /home/yehoon/workspace/llm_a_to_z/llama_2/ft_llama2.py\
  --model="meta-llama/Llama-2-7b-hf"\
  --dataset="databricks/databricks-dolly-15k"\
  --output_dir="/home/yehoon/workspace/outputs/"\
  --output_name="0727_llama2_on_dolly_15k"\
  --lora_r=8\
  --lora_alpha=16\
  --lora_dropout=0.05\
  --batch_size=1\
  --epoch=3\
  --gradient_step=1\
  --warmup=2\
  --lr=5e-4\
  --optimizer="paged_adamw_8bit"\
  --max_memory=10240\
  --device_map="auto"\
  --seed=42

