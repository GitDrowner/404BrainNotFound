from __future__ import annotations

import json

import PIL
import torch
import torchvision


def main() -> None:
    import timm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the Slurm allocation")
    inputs = torch.randn(2, 3, 64, 64, device="cuda")
    layer = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
    outputs = layer(inputs)
    torch.cuda.synchronize()
    payload = {
        "status": "PASS",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "pillow": PIL.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "conv_shape": list(outputs.shape),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
