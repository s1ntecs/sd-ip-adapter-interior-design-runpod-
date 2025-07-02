# download_checkpoints.py  (offline-build)

import os
import torch

from diffusers import ControlNetModel
from diffusers.pipelines.controlnet import StableDiffusionControlNetInpaintPipeline
from diffusers import ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import MLSDdetector
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from huggingface_hub import hf_hub_download, snapshot_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LORA_NAMES = [
    "XSArchi_110plan彩总.safetensors",
    "XSArchi_137.safetensors",
    "XSArchi_141.safetensors",
    "XSArchi_162BIESHU.safetensors",
    "XSarchitectural-38InteriorForBedroom.safetensors",
    "XSarchitectural_33WoodenluxurystyleV2.safetensors",
    "house_architecture_Exterior_SDlife_Chiasedamme.safetensors",
    "xsarchitectural-15Nightatmospherearchitecture.safetensors",
    "xsarchitectural-18Whiteexquisiteinterior.safetensors",
    "xsarchitectural-19Houseplan (1).safetensors",
    "xsarchitectural-19Houseplan.safetensors",
    "xsarchitectural-7.safetensors"
]


# ------------------------- загрузка весов -------------------------
def fetch_checkpoints() -> None:
    """Скачиваем SD-чекпойнт, LoRA-файлы и всё нужное для IP-Adapter."""
    # сами веса адаптера
    hf_hub_download(
        repo_id="h94/IP-Adapter",
        subfolder="models",
        filename="ip-adapter_sd15.bin",
        local_dir="./ip_adapter",
        local_dir_use_symlinks=False,
    )

    # обязательный CLIP-Vision энкодер ─ без него load_ip_adapter упадёт
    hf_hub_download(
        repo_id="h94/IP-Adapter",
        subfolder="models/image_encoder",
        filename="config.json",
        local_dir="./ip_adapter",
        local_dir_use_symlinks=False,
    )
    hf_hub_download(
        repo_id="h94/IP-Adapter",
        subfolder="models/image_encoder",
        filename="model.fp16.safetensors",
        local_dir="./ip_adapter",
        local_dir_use_symlinks=False,
    )

    # все ваши LoRA-файлы
    for fname in LORA_NAMES:
        hf_hub_download(
            repo_id="sintecs/interior",
            filename=fname,
            local_dir="loras",
            local_dir_use_symlinks=False,
        )


# ------------------------- пайплайн -------------------------
def get_pipeline():
    controlnet = [
        ControlNetModel.from_pretrained(
            "BertChristiaens/controlnet-seg-room", torch_dtype=torch.float16
        ),
        ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
        ),
    ]
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V3.0_VAE",
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16,
        )

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    pipe.load_ip_adapter(
        "./ip_adapter",
        subfolder="models",
        weight_name="ip-adapter_sd15.bin",
    )

    # pipe.enable_xformers_memory_efficient_attention()
    # pipe = pipe.to(DEVICE)
    AutoImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    MLSDdetector.from_pretrained("lllyasviel/Annotators")

    return pipe


if __name__ == "__main__":
    fetch_checkpoints()
    get_pipeline()
