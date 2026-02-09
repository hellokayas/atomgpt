#!/usr/bin/env python3
import os
# ---- Disable TorchDynamo/torch.compile globally (pre-Torch import is safest) ----
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # cleanest for inference
# Optional belt-and-suspenders for some stacks:
os.environ.setdefault("PYTORCH_DISABLE_JIT", "1")

import re
import torch
from PIL import Image
from peft import PeftModel
from atomgpt.inverse_models.loader import FastVisionModel

# If Dynamo still gets touched by a wrapper somewhere:
try:
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

# ----------------------------------------------------------------------------------------------------------------------
# CONFIG: Models and checkpoints
# ----------------------------------------------------------------------------------------------------------------------
BASE_MODELS = {
    "gemma": "/projects/p32726/microscopy-gpt/atomgpt/atomgpt/models/unsloth/gemma-3-12b-it-bnb-4bit",
    "llama": "/projects/p32726/microscopy-gpt/atomgpt/atomgpt/models/unsloth/Llama-3.2-11B-Vision-Instruct",
    "qwen":  "/projects/p32726/microscopy-gpt/atomgpt/atomgpt/models/unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit",
}

CKPTS = [
    {"tag": "gemma_dft2d", "base_key": "gemma",
     "ckpt_path": "/projects/p32726/microscopy-gpt/atomgpt/atomgpt/formula_based_dft_2d_unsloth_old_prompt/gemma-3-12b-it-bnb-4bit/checkpoint-992"},
    {"tag": "llama_dft2d", "base_key": "llama",
     "ckpt_path": "/projects/p32726/microscopy-gpt/atomgpt/atomgpt/formula_based_dft_2d_unsloth_old_prompt/Llama-3.2-11B-Vision-Instruct/checkpoint-1240"},
    {"tag": "qwen_dft2d",  "base_key": "qwen",
     "ckpt_path": "/projects/p32726/microscopy-gpt/atomgpt/atomgpt/formula_based_dft_2d_unsloth_old_prompt/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit/checkpoint-1240"},
    {"tag": "gemma_c2db",  "base_key": "gemma",
     "ckpt_path": "/projects/p32726/microscopy-gpt/atomgpt/atomgpt/formula_based_c2db/gemma-3-12b-it-bnb-4bit/checkpoint-4334"},
    {"tag": "llama_c2db",  "base_key": "llama",
     "ckpt_path": "/projects/p32726/microscopy-gpt/atomgpt/atomgpt/formula_based_c2db/Llama-3.2-11B-Vision-Instruct/checkpoint-4334"},
    {"tag": "qwen_c2db",   "base_key": "qwen",
     "ckpt_path": "/projects/p32726/microscopy-gpt/atomgpt/atomgpt/formula_based_c2db/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit/checkpoint-4334"},
]

# ----------------------------------------------------------------------------------------------------------------------
# CONFIG: Images & prompts (extensible)
# Add/modify entries below. Each task has an image path and a formula.
# You can keep appending tasks to this list without changing any other code.
# ----------------------------------------------------------------------------------------------------------------------
IMG_DIR = "/projects/p32726/microscopy-gpt/atomgpt/atomgpt/formula_based"

TASKS = [

    # {"name": "a", "image": f"{IMG_DIR}/a.png", "formula": "C"},
    # {"name": "b", "image": f"{IMG_DIR}/b.png", "formula": "C"},
    # {"name": "c", "image": f"{IMG_DIR}/c.png", "formula": "FeTe"},
    # {"name": "d", "image": f"{IMG_DIR}/d.png", "formula": "FeTe"},

    {"name": "FeTe", "image": f"{IMG_DIR}/tetra-FeTe.png", "formula": "FeTe"},
    {"name": "C",    "image": f"{IMG_DIR}/exp-C.png",    "formula": "C"},
    {"name": "FeTe", "image": f"{IMG_DIR}/hex-FeTe.png", "formula": "FeTe"},
    # {"name": "MoS2", "image": f"{IMG_DIR}/MoS2.png","formula": "MoS2"},
]


# ----------------------------------------------------------------------------------------------------------------------
# Prompt template
# ----------------------------------------------------------------------------------------------------------------------
PROMPT_TEMPLATE = (
    "The chemical formula is {formula}. Generate atomic structure description with lattice lengths, "
    "angles, coordinates, and atom types. Also predict the Miller index.. Your primary constraint: "
    "**do not change the stoichiometry**. Use exactly the same elements and exactly the same "
    "integer counts as in the formula above. Do not add, remove, merge, reduce, or simplify elements. "
    "If the formula is Br9Nb2Rb2, your output must still contain Br (9), Nb (2), and Rb (2)."
)

def make_prompt(formula: str) -> str:
    return PROMPT_TEMPLATE.format(formula=formula)

# ----------------------------------------------------------------------------------------------------------------------
# Generation parameters
# ----------------------------------------------------------------------------------------------------------------------
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0
TOP_P = 1.0

# ----------------------------------------------------------------------------------------------------------------------
# Loading / inference helpers
# ----------------------------------------------------------------------------------------------------------------------
def load_model_with_adapter(base_model_dir: str, adapter_dir: str, device: str = "cuda"):
    base_model, tokenizer = FastVisionModel.from_pretrained(
        base_model_dir,
        load_in_4bit=True,
        use_gradient_checkpointing=False,  # no need for inference
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    model.to(device)
    return model, tokenizer

def build_inputs(tokenizer, image: Image.Image, text_prompt: str, device: str):
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": text_prompt}]}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt")
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

MODEL_ANSWER_RE = re.compile(r"<start_of_turn>model(.*?)(?:<end_of_turn>|$)", re.DOTALL)

def extract_answer(decoded: str) -> str:
    m = MODEL_ANSWER_RE.search(decoded)
    if m:
        return m.group(1).strip()
    cleaned = decoded.replace("<bos>", "").replace("<eos>", "")
    cleaned = cleaned.replace("<start_of_turn>user", "").replace("<start_of_turn>model", "")
    cleaned = cleaned.replace("<end_of_turn>", "")
    cleaned = cleaned.replace("<start_of_image>", "").replace("<end_of_image>", "")
    cleaned = cleaned.replace("<image_soft_token>", "")
    return cleaned.strip()

def generate_once(model, tokenizer, image_path: str, prompt: str, device: str):
    img = Image.open(image_path).convert("RGB")
    inputs = build_inputs(tokenizer, img, prompt, device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(TEMPERATURE > 0),
            temperature=TEMPERATURE,
            top_p=TOP_P,
            use_cache=True,
        )
    decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    return extract_answer(decoded)

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    print(f"Device: {device}")
    print(f"Starting inference for {len(CKPTS)} checkpoints × {len(TASKS)} image(s)...\n")

    for spec in CKPTS:
        tag, base_key, adapter = spec["tag"], spec["base_key"], spec["ckpt_path"]
        base_dir = BASE_MODELS[base_key]
        print("=" * 120)
        print(f"[MODEL] {tag}\n  Base:   {base_dir}\n  Adapter:{adapter}")
        try:
            model, tokenizer = load_model_with_adapter(base_dir, adapter, device)
        except Exception as e:
            print(f"  ! Load failed: {e}")
            continue

        any_success = False
        for task in TASKS:
            name = task.get("name", "unnamed")
            img_path = task["image"]
            prompt = task.get("custom_prompt") or make_prompt(task["formula"])

            print("-" * 120)
            print(f"[INPUT] name={name} | image={img_path} | formula={task['formula']}")
            try:
                answer = generate_once(model, tokenizer, img_path, prompt, device)
                print("[ANSWER]")
                print(answer)
                any_success = True
            except Exception as e:
                print(f"  ! Generation failed: {e}")

        # Free memory between models
        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

        if not any_success:
            print("  → No successful generations for this model.")

    print("\nDone.")

if __name__ == "__main__":
    os.environ["AtomGPT_RETURN_LOGITS"] = "1"
    main()
