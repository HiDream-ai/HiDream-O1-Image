# HiDream-O1-Image-Dev-2604


## Models

| Name | Script | Inference Steps | HuggingFace Repo |
| :--- | :--- | :---: | :--- |
| HiDream-O1-Image-Dev-2604 | `inference.py` | 28 | [🤗 HiDream-O1-Image-Dev-2604](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev-2604) |
| Prompt Agent | `prompt_agent_v2.py` | — | [🤗 HiDream-ai/gemma-4-31B-it-Prompt-Refine](https://huggingface.co/HiDream-ai/gemma-4-31B-it-Prompt-Refine) |

## Installation

1. Clone this repository:
```bash
git clone https://github.com/HiDream-ai/HiDream-O1-Image.git
cd HiDream-O1-Image
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

> **Note on `flash-attn`.** We highly recommend installing [`flash-attn`](https://github.com/Dao-AILab/flash-attention) for optimized attention computation. **If you do not (or cannot) install `flash-attn`, you must edit `models/pipeline.py` line 291 and change `"use_flash_attn": True` to `"use_flash_attn": False`** — otherwise inference will fail to import the kernel.

## Reasoning-Driven Prompt Agent

HiDream-O1-Image ships with a Reasoning-Driven Prompt Agent (`prompt_agent_v2.py`) that explicitly reasons through layout, subject attributes, physical logic, and text-rendering details, then rewrites a raw user instruction into a self-contained English prompt. Feed its output into `inference.py` for best results on intricate, reasoning-heavy requests.

The agent talks to an OpenAI-compatible endpoint serving [`HiDream-ai/gemma-4-31B-it-Prompt-Refine`](https://huggingface.co/HiDream-ai/gemma-4-31B-it-Prompt-Refine) via vLLM.

### Step 1 — Download the refiner weights

```bash
huggingface-cli download HiDream-ai/gemma-4-31B-it-Prompt-Refine \
    --local-dir HiDream-ai/gemma-4-31B-it-Prompt-Refine
```

### Step 2 — Start the vLLM server

```bash
bash start_vllm_server.sh
```

This launches `HiDream-ai/gemma-4-31B-it-Prompt-Refine` on `http://localhost:8000/v1`.

### Step 3 — Run the refiner

```bash
python prompt_agent_v2.py \
    --prompt "李白的静夜思写在古墙上"
```

By default the script targets `http://localhost:8000/v1` and `HiDream-ai/gemma-4-31B-it-Prompt-Refine`; override with `--base_url` or `--model_id` if you serve the model elsewhere. The same module also exposes a reusable `refine_prompt(prompt, model_id=..., base_url=...)` function used by `app.py`.

## Usage

A CUDA-capable GPU is required for inference. The examples below use the **undistilled** model (`--model_type full`); see the last subsection for running the same tasks with the **distilled** model (`--model_type dev`).

### 1. Text-to-Image Generation
Generate an image from a text prompt:

```bash
python inference.py \
    --model_path /path/to/HiDream-O1-Image \
    --prompt "medium shot, eye-level, front view. A woman is seated in an ornate bedroom, illuminated by candlelight, with a calm and composed expression. The subject is a young woman with fair skin, light brown hair styled in an updo with loose tendrils framing her face, and blue eyes. She wears a cream-colored satin robe with delicate floral embroidery and lace trim along the neckline. Her ears are adorned with pearl drop earrings. She is seated on a bed with a dark, intricately carved wooden headboard. To her left, a wooden nightstand holds three lit white candles and a candelabra with multiple lit candles in the background. The bed is covered with patterned pillows and a dark, textured blanket. The walls are paneled with dark wood and feature a large, ornate tapestry with muted earth tones. The lighting creates soft highlights on her face and robe, with warm shadows cast across the room." \
    --output_image results/t2i.png \
    --height 2048 \
    --width 2048
```

## License
The code in this repository and the HiDream-O1-Image models are licensed under [MIT License](./LICENSE).
