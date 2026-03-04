"""Fix vLLM nightly bug in qwen3_5_moe config.

ignore_keys_at_rope_validation must be a set for the | operator used in
transformers rope validation, but must be a list for JSON serialization.
"""
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/qwen3_5_moe.py"
text = open(path).read()

# Change list literal [...] to set literal {...}
old = '"ignore_keys_at_rope_validation"] = [\n            "mrope_section",\n            "mrope_interleaved",\n        ]'
new = '"ignore_keys_at_rope_validation"] = {\n            "mrope_section",\n            "mrope_interleaved",\n        }'
text = text.replace(old, new)

# After super().__init__(), convert set back to list for JSON serialization
old_super = "super().__init__(**kwargs)\n"
new_super = (
    "super().__init__(**kwargs)\n"
    "        if hasattr(self, \"ignore_keys_at_rope_validation\") and isinstance(self.ignore_keys_at_rope_validation, set):\n"
    "            self.ignore_keys_at_rope_validation = list(self.ignore_keys_at_rope_validation)\n"
)
text = text.replace(old_super, new_super)

open(path, "w").write(text)
print("Patched qwen3_5_moe.py")
