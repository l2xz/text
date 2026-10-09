# encoding=utf-8
import os
from transformers import AutoTokenizer

def fix_and_save_tokenizer(model_path):
    print(f">>> [Tokenizer Tool] Fixing tokenizer at: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            fix_mistral_regex=True
        )

        tokenizer.save_pretrained(model_path)
        print(f">>> [Tokenizer Tool] Success: Tokenizer configuration updated.")
        return True

    except Exception as e:
        print(f">>> [Tokenizer Tool] Error fixing tokenizer: {e}")
        return False