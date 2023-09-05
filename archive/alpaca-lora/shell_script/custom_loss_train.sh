python finetune_custom_loss.py \
--base_model 'decapoda-research/llama-7b-hf' \
--data_path "../data/ko_alpaca_data.json" \
--output_dir "../outputs/alpaca-lora/llama_7b_koalpaca_custom_loss" \
--num_epochs 3 \
--learning_rate 5e-4 \
--val_set_size 2000 \
--batch_size 8 \
--micro_batch_size 1 \
--prompt_template_name "custom"
