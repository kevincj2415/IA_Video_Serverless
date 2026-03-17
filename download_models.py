import os
from huggingface_hub import snapshot_download, hf_hub_download

def download():
    hf_token = os.environ.get("HF_TOKEN")
    hf_token = hf_token.strip() if hf_token else None
    
    base_model = os.environ.get("BASE_MODEL", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    vae_model = os.environ.get("VAE_MODEL", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    lora_model = os.environ.get("LORA_MODEL", "wangkanai/wan22-fp8-i2v-loras-nsfw")
    lora_filename = os.environ.get("LORA_FILENAME", "loras/wan/wan22-action-missionary-pov-i2v-low.safetensors")

    print(f"Buscando / Descargando Base Model: {base_model}...")
    snapshot_download(repo_id=base_model, token=hf_token)

    print(f"Buscando / Descargando VAE: {vae_model}...")
    snapshot_download(repo_id=vae_model, allow_patterns=["vae/*"], token=hf_token)

    print(f"Buscando / Descargando LoRA: {lora_model} / {lora_filename}...")
    hf_hub_download(repo_id=lora_model, filename=lora_filename, token=hf_token)
    
    print("Modelos descargados exitosamente en la cache de Hugging Face.")

if __name__ == "__main__":
    download()
