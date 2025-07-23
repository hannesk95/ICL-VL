# llava_folder_test.py  –  role-tagged, deterministic, single-GPU

import pathlib, textwrap, torch, PIL.Image as Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

MODEL_ID = "llava-hf/llava-1.5-7b-hf"          # swap for another checkpoint if you like
IMG_DIR  = pathlib.Path("demo_imgs")            # put your .jpg / .png files here

SYSTEM_PROMPT = "You are a factual visual assistant. Be concise."
QUESTION      = "Describe the image in detail. What is the main landmark or feature?"

# 1) collect images
imgs = sorted(p for p in IMG_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
if not imgs:
    raise SystemExit(f"No .jpg/.png found in {IMG_DIR.resolve()}")
print(f"Found {len(imgs)} images in {IMG_DIR}")

# 2) load model (4-bit on cuda:0)
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
proc  = AutoProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID, quantization_config=bnb_cfg, device_map={"": 0}
        ).eval()

# 3) iterate
for path in imgs:
    img = Image.open(path).convert("RGB")

    # LLaVA chat template expects exactly this structure
    prompt = (
        f"SYSTEM: {SYSTEM_PROMPT}\n"
        f"USER: <image>\n{QUESTION}\n"
        f"ASSISTANT:"
    )

    print(f"\n🖼️  {path.name} — Q: {QUESTION}")

    inputs = proc(text=prompt, images=[img], return_tensors="pt").to("cuda:0")
    out_ids = model.generate(**inputs, do_sample=False, max_new_tokens=128)
    answer  = proc.decode(out_ids[0], skip_special_tokens=True).strip()

    # LLaVA sometimes repeats the assistant tag at the start; strip it
    answer = answer.lstrip("ASSISTANT:").strip()
    print(textwrap.fill(answer, width=90))