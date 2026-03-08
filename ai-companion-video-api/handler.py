import os
import runpod
import torch
import base64
import tempfile
from io import BytesIO
from PIL import Image

from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.utils import export_to_video

hf_token = os.environ.get("HF_TOKEN")

# Model paths or HF repo IDs (Allow overrides for network volumes in RunPod)
base_model = os.environ.get("BASE_MODEL", "Wan-AI/Wan2.2-I2V-14B")
vae_model = os.environ.get("VAE_MODEL", "Wan-AI/Wan2.2-VAE")
lora_model = os.environ.get("LORA_MODEL", "wangkanai/wan22-fp8-i2v-loras-nsfw")
lora_filename = os.environ.get("LORA_FILENAME", "wan22-action-missionary-pov-i2v-low.safetensors")

print(f"Loading Base Model: {base_model}")

device = "cuda"
dtype = torch.bfloat16

# Load pipeline
pipe = DiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=dtype,
    token=hf_token
)

# Load VAE
if vae_model.endswith(".safetensors"):
    pipe.vae = AutoencoderKL.from_single_file(vae_model, torch_dtype=dtype, token=hf_token)
else:
    pipe.vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=dtype, token=hf_token)

# Load LoRA
print(f"Loading LoRA: {lora_model} / {lora_filename}")
pipe.load_lora_weights(lora_model, weight_name=lora_filename, token=hf_token)

# Memory optimization for lower VRAM
pipe.enable_model_cpu_offload()

def handler(job):
    job_input = job.get("input", {})
    
    # Extract parameters with defaults based on documentation
    prompt = job_input.get("prompt", "POV perspective, smooth natural movement, realistic, high quality")
    image_base64 = job_input.get("image")
    num_inference_steps = job_input.get("num_inference_steps", 50) # 40-60 steps optimal
    guidance_scale = job_input.get("guidance_scale", 7.5)          # 7.0-8.5 for controlled motion
    num_frames = job_input.get("num_frames", 24)                   # 16-32 frames supported
    height = job_input.get("height", 720)                          # 720p optimized
    width = job_input.get("width", 1280)
    fps = job_input.get("fps", 8)
    seed = job_input.get("seed", 0)

    if not image_base64:
        return {"error": "Missing input image (base64 encoded) in the 'image' field"}

    try:
        # Decode input image
        image_data = base64.b64decode(image_base64)
        input_image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Generator for reproducibility
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

        # Generate video
        video_frames = pipe(
            prompt=prompt,
            image=input_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            height=height,
            width=width,
            generator=generator
        ).frames

        # Save video to temporary file
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"output_{job['id']}.mp4")
        
        # export_to_video receives the video frames
        export_to_video(video_frames, output_path, fps=fps)

        # Read file and encode to base64
        with open(output_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)

        return {"video_base64": video_base64}
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
