from transformers import MistralConfig, MixtralConfig
import json

config = MixtralConfig(
    vocab_size=25,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=16,
    num_attention_heads=16,
    num_key_value_heads=4,
    head_dim=64,
    pad_token_id=0,
    max_position_embeddings=8192,
    hidden_act="silu",
    rms_norm_eps=1e-5,
    rope_theta=1e6,
    attention_dropout=0.0,
    initializer_range=0.02,
    sliding_window=None,
    use_cache=True,
    tie_word_embeddings=False,
    torch_dtype="bfloat16"
)

config.save_pretrained("mixtral")  # 会保存为 ./configs/mistral_config/config.json
