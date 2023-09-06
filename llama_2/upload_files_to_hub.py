from huggingface_hub import HfApi

files = ["special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "training_args.bin"]

api = HfApi()
for file_name in files:
    api.upload_file(
        path_or_fileobj=f"../../outputs/0905_ft_llama2_upstage_way/{file_name}",
        path_in_repo=file_name,
        repo_id="Yehoon/yehoon_llama2",
        repo_type="model"

    )