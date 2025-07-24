from transformers import AutoProcessor, LlavaForConditionalGeneration

print("Downloading and loading Microsoft Med‑LLaVA v1.5 Mistral‑7B...")

processor = AutoProcessor.from_pretrained(
    "microsoft/llava-med-v1.5-mistral-7b",
    trust_remote_code=True
)
model = LlavaForConditionalGeneration.from_pretrained(
    "microsoft/llava-med-v1.5-mistral-7b",
    trust_remote_code=True
)

print("✅ Model loaded successfully!")