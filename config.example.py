CONFIG = {
    "data_dir": "data/raw",
    "processed_dir": "data/processed",
    "modality": "solid_state",
    "iterations": 250,
    "top_k": 5,
    "exploration_constant": 1.4,
    "rollout_count": 8,
    "seed": 0,
    "judge": {
        "name": "openai_structured",
        "model": "gpt-oss-120b",
        "api_key": "paste-your-api-key-here",
        # Optional for OpenAI-compatible providers or self-hosted endpoints.
        "base_url": "https://aiportal-api.aws.lanl.gov",
        "api_style": "chat_completions",
    },
}
