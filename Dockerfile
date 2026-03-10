FROM pytorch/pytorch:2.8.1-cuda12.6-cudnn12-devel

WORKDIR /app

# Instalar dependencias del sistema requeridas para video (ffmpeg) y git
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

# Instalar diffusers y dependencias necesarias para WAN 2.2 I2V
# Se usan las mismas que la documentación
RUN pip install git+https://github.com/huggingface/diffusers.git transformers accelerate safetensors sentencepiece protobuf runpod imageio[ffmpeg] --upgrade --break-system-packages

COPY handler.py .

CMD [ "python", "-u", "handler.py" ]
