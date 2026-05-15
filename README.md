# HiDream-O1-Image-Dev-2604 for Text-to-Image Generation


## Models

| Name | Script | Inference Steps | HuggingFace Repo |
| :--- | :--- | :---: | :--- |
| HiDream-O1-Image-Dev-2604 | `inference.py` | 28 | [🤗 HiDream-O1-Image-Dev-2604](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev-2604) |
| Prompt Agent 2604 | `prompt_agent_v2.py` | — | [🤗 HiDream-ai/Prompt-Refine](https://huggingface.co/HiDream-ai/Prompt-Refine) |

## Installation

1. Clone this repository:
```bash
git clone https://github.com/HiDream-ai/HiDream-O1-Image.git
cd HiDream-O1-Image
git checkout dev
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

> **Note on `flash-attn`.** We highly recommend installing [`flash-attn`](https://github.com/Dao-AILab/flash-attention) for optimized attention computation. **If you do not (or cannot) install `flash-attn`, you must edit `models/pipeline.py` line 291 and change `"use_flash_attn": True` to `"use_flash_attn": False`** — otherwise inference will fail to import the kernel.

## Reasoning-Driven Prompt Agent

HiDream-O1-Image ships with a Reasoning-Driven Prompt Agent (`prompt_agent_v2.py`) that explicitly reasons through layout, subject attributes, physical logic, and text-rendering details, then rewrites a raw user instruction into a self-contained English prompt. Feed its output into `inference.py` for best results on intricate, reasoning-heavy requests.

The agent talks to an OpenAI-compatible endpoint serving [`HiDream-ai/Prompt-Refine`](https://huggingface.co/HiDream-ai/Prompt-Refine) via vLLM.

### Step 1 — Download the refiner weights

```bash
huggingface-cli download HiDream-ai/Prompt-Refine \
    --local-dir HiDream-ai/Prompt-Refine
```

### Step 2 — Start the vLLM server

```bash
bash start_vllm_server.sh
```

This launches `HiDream-ai/Prompt-Refine` on `http://localhost:8000/v1`.

### Step 3 — Run the refiner

```bash
python prompt_agent_v2.py \
    --prompt "A vintage aviation poster featuring a bright red biplane cruising over rolling farmlands. Bold blocky text at the bottom promises adventure in the friendly skies."
```

By default the script targets `http://localhost:8000/v1` and `HiDream-ai/Prompt-Refine`; override with `--base_url` or `--model_id` if you serve the model elsewhere. The same module also exposes a reusable `refine_prompt(prompt, model_id=..., base_url=...)` function used by `app.py`.

## Usage

A CUDA-capable GPU is required for inference. The examples below use the **undistilled** model (`--model_type full`); see the last subsection for running the same tasks with the **distilled** model (`--model_type dev`).

### 1. Text-to-Image Generation
Generate an image from a text prompt:

```bash
python inference.py \
    --model_path /path/to/HiDream-O1-Image-Dev-2604 \
    --prompt "A vintage aviation poster depicting a bright red biplane cruising over rolling farmlands under a partly cloudy sky, with saturated colors and an aged paper texture. A red biplane with two sets of wings and a radial engine is positioned in the upper center of the image, flying toward the right. A pilot with light skin, wearing a brown flight helmet, goggles, and a brown jacket, is visible in the open cockpit. The biplane has black wheels with red hubs and a spinning propeller. Below, the landscape consists of rolling fields in various shades of green, yellow, and brown, divided by dirt roads and scattered with small houses, including a red barn, a brown house, and a white house. In the background, a line of green trees separates the fields from distant hills under a blue sky with white clouds. The poster has a textured, aged paper border with visible creases and discoloration. At the bottom, the text \"ADVENTURE IN THE FRIENDLY SKIES\" is displayed in large, bold, dark brown capital letters across two lines on a light beige background." \
    --output_image results/t2i.png \
    --height 2048 \
    --width 2048
```

## License
The code in this repository and the HiDream-O1-Image-Dev-2604 models are licensed under [MIT License](./LICENSE).
