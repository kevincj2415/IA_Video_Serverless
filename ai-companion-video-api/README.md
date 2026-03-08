# AI Companion Video API (WAN 2.2 I2V)

Este es el endpoint Serverless para la generación de Video a partir de Imagen (I2V) usando **WAN 2.2 14B** y el LoRA de acción específico. Está diseñado para ejecutarse en **RunPod Serverless**.

## Características

- Basado en el modelo WAN 2.2 I2V 14B.
- Soporta VAE separado (`wan22-vae`).
- Integra el LoRA de acción: `wangkanai/wan22-fp8-i2v-loras-nsfw` (`wan22-action-missionary-pov-i2v-low.safetensors`).
- Aceleración con `enable_model_cpu_offload()` para ahorro de VRAM.

## Requisitos de Hardware (RunPod)

- **GPU Recomendada:** RTX 3090, RTX 4070 Ti, RTX 4090 o A100.
- **Volumen de Red:** Dado el tamaño del modelo base (14GB+), se recomienda encarecidamente utilizar un **Network Volume** de RunPod para almacenar los modelos y evitar que se descarguen en cada "Cold Start".

## Variables de Entorno (Environment Variables)

Configura las siguientes variables en la plantilla de RunPod:

- `HF_TOKEN`: Tu token de Hugging Face (requerido si usas repos privados).
- `BASE_MODEL`: El modelo base de difusión (por defecto: `Wan-AI/Wan2.2-I2V-14B`).
- `VAE_MODEL`: El modelo VAE (por defecto: `Wan-AI/Wan2.2-VAE`).
- `LORA_MODEL`: Repositorio o carpeta local del LoRA (por defecto: `wangkanai/wan22-fp8-i2v-loras-nsfw`).
- `LORA_FILENAME`: Nombre del archivo SafeTensor del LoRA (por defecto: `wan22-action-missionary-pov-i2v-low.safetensors`).

**Nota:** Si guardas los modelos en un volumen de red (ej. `/runpod-volume`), cambia estas variables por las rutas absolutas dentro del contenedor para evitar descargas innecesarias.

## Ejemplo de JSON Payload

```json
{
  "input": {
    "prompt": "POV perspective, smooth natural movement, cinematic lighting, high quality, realistic, 720p",
    "image": "iVBORw0KGgoAAAANSUh...", 
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "num_frames": 24,
    "height": 720,
    "width": 1280,
    "fps": 8,
    "seed": 42
  }
}
```

- **`image`**: (*Requerido*) Cadena Base64 de la imagen inicial a animar.
- **`prompt`**: (*Opcional*) Prompt descriptivo para la acción.
- **`num_inference_steps`**: (*Opcional*) 40-60 pasos óptimos.
- **`guidance_scale`**: (*Opcional*) Recomendado 7.0 - 8.5.
- **`num_frames`**: (*Opcional*) Recomendado 16-32.
- **`height` / `width`**: (*Opcional*) Modifican la resolución (ej: 720x1280).

## Respuesta

Retornará un objeto JSON con el video generado en formato Base64:

```json
{
  "video_base64": "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMQ..."
}
```

## Despliegue en RunPod Serverless

1. Construye y sube la imagen Docker a un registro como Docker Hub:

   ```bash
   docker build -t tu_usuario/ai-video-api:latest .
   docker push tu_usuario/ai-video-api:latest
   ```

2. Crea un **Serverless Endpoint** en RunPod.
3. Selecciona tu imagen `tu_usuario/ai-video-api:latest`.
4. Añade tus variables de entorno.
5. Despliega y obtén el ID/URL de tu endpoint de la API.
