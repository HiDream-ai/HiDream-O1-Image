import argparse
import torch
import os
from transformers import AutoProcessor
from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
from models.pipeline import generate_image, DEFAULT_TIMESTEPS

def add_special_tokens(tokenizer):
    """Attach the special-token shortcuts that the pipeline relies on."""
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"

def get_tokenizer(processor):
    from transformers import PreTrainedTokenizerBase
    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer

def main():
    p = argparse.ArgumentParser("Standalone Pixel-DiT inference")
    p.add_argument("--model_path", type=str, default = "/root/converted_models/HiDream-O1-Image", help="Path to huggingface model")
    p.add_argument("--prompt", type=str, default = "medium shot, eye-level, front view. A woman is seated in an ornate bedroom, illuminated by candlelight, with a calm and composed expression. The subject is a young woman with fair skin, light brown hair styled in an updo with loose tendrils framing her face, and blue eyes. She wears a cream-colored satin robe with delicate floral embroidery and lace trim along the neckline. Her ears are adorned with pearl drop earrings. She is seated on a bed with a dark, intricately carved wooden headboard. To her left, a wooden nightstand holds three lit white candles and a candelabra with multiple lit candles in the background. The bed is covered with patterned pillows and a dark, textured blanket. The walls are paneled with dark wood and feature a large, ornate tapestry with muted earth tones. The lighting creates soft highlights on her face and robe, with warm shadows cast across the room.")
    p.add_argument("--ref_images", nargs="*", default=[], help="Path to reference images.")
    p.add_argument(
        "--layout_bboxes",
        type=str,
        default=None,
        help=(
            "Layout boxes as a JSON string or JSON file path. "
            "Input order is xxyy relative coordinates: [[x1,x2,y1,y2], ...], "
            "[{\"bbox\": [x1,x2,y1,y2], \"text\": \"...\"}, ...], "
            "or {\"bboxes\": ...}. Example: [0.1,0.15,0.2,0.36]."
        ),
    )
    p.add_argument("--output_image", type=str, default="output.png")
    p.add_argument("--height", type=int, default=2048)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--model_type", type=str, default="dev", choices=["full", "dev"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--noise_scale_start", type=float, default=8.0)
    p.add_argument("--noise_scale_end", type=float, default=8.0)
    p.add_argument("--noise_clip_std", type=float, default=8.0)
    p.add_argument(
        "--editing_scheduler",
        type=str,
        default="flow_match",
        choices=["flow_match", "flash"],
        help="Scheduler used for editing (exactly one reference image) when "
             "--model_type dev. Default: flow_match. Ignored for full model "
             "and for non-editing tasks.",
    )
    p.add_argument(
        "--keep_original_aspect",
        action="store_true",
        help="When exactly one reference image is provided, resize it with "
             "max_size=2048 and use its dimensions for the target image "
             "(preserves the reference's aspect ratio).",
    )
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA is required for inference."
    print(f"[inference] Loading processor and model from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    # NOTE: torch_dtype = torch.float32 will generate more detailed images but with more memory usage
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float32, device_map="cuda"
    ).eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_image)), exist_ok=True)
    tokenizer = get_tokenizer(processor)
    add_special_tokens(tokenizer)

    extra_kwargs = {}
    if args.model_type == "full":
        num_inference_steps = 50
        guidance_scale = args.guidance_scale
        shift = args.shift
        timesteps_list = None
        scheduler_name = "default"
    else:
        is_editing = len(args.ref_images) == 1
        if is_editing and args.editing_scheduler == "flow_match":
            num_inference_steps = 28
            guidance_scale = 0.0
            shift = 1.0
            timesteps_list = DEFAULT_TIMESTEPS
            scheduler_name = "flow_match"
        else:
            num_inference_steps = 28
            guidance_scale = 0.0
            shift = 1.0
            timesteps_list = DEFAULT_TIMESTEPS
            scheduler_name = "flash"
            extra_kwargs["noise_scale_start"] = args.noise_scale_start
            extra_kwargs["noise_scale_end"] = args.noise_scale_end
            extra_kwargs["noise_clip_std"] = args.noise_clip_std

    image = generate_image(
        model=model,
        processor=processor,
        prompt=args.prompt,
        ref_image_paths=args.ref_images,
        height=args.height,
        width=args.width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        shift=shift,
        timesteps_list=timesteps_list,
        scheduler_name=scheduler_name,
        seed=args.seed,
        keep_original_aspect=args.keep_original_aspect,
        layout_bboxes = args.layout_bboxes,
        **extra_kwargs,
    )
    image.save(args.output_image)
    print(f"[inference] Saved -> {args.output_image}")

if __name__ == "__main__":
    main()
