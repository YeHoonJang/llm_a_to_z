#python finetune.py \
#--base_model 'decapoda-research/llama-7b-hf' \
#--data_path "data/ko_alpaca_style_dataset.json" \
#--output_dir "./outputs/llama_7b_ shareGPT_kor" \
#--num_epochs 3 \
#--learning_rate 5e-4 \
#--val_set_size 5000 \
#--batch_size 256 \
#--micro_batch_size 32 \
#--prompt_template_name "custom"

python finetune.py \
--base_model 'decapoda-research/llama-7b-hf' \
--data_path "../data/alpaca_data.json" \
--output_dir "../outputs/llama_7b_alpaca" \
--num_epochs 3 \
--learning_rate 5e-4 \
--val_set_size 2000 \
--batch_size 128 \
--micro_batch_size 16 \
--prompt_template_name "custom"


#python finetune.py \
#--base_model './outputs/llama_7b_koalpaca/checkpoint-200' \
#--data_path "data/ko_alpaca_data.json" \
#--output_dir "./outputs/llama_7b_koalpaca" \
#--num_epochs 3 \
#--learning_rate 5e-4 \
#--val_set_size 5000 \
#--batch_size 512 \
#--micro_batch_size 64 \
#--prompt_template_name "custom"
