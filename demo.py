



import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from demo_util import get_titok_tokenizer
import matplotlib.pyplot as plt

device = torch.device("cuda:0")
# model_type = "titok" # "optvq"
model_type = "optvq" # "optvq"

# load the tokenizer
if model_type == "titok":
    config = OmegaConf.load("./configs/titok_maskgit.yaml")
    config.experiment.tokenizer_checkpoint = "huggingface:yucornetto/tokenizer_titok_b64_imagenet"
elif model_type == "optvq":
    config = OmegaConf.load("./configs/optvq_maskgit.yaml")
    config.experiment.tokenizer_checkpoint = "./outputs/optvq_b64_stage2_run5-dinodisc/checkpoint-1000000/ema_model/pytorch_model.bin"
tokenizer = get_titok_tokenizer(config)
tokenizer.to(device)

img_path = "assets/ILSVRC2012_val_00008636.png"
image = torch.from_numpy(np.array(Image.open(img_path)).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
image = image.to(device)






with torch.no_grad():
    tokens = tokenizer.encode(image)[1]["min_encoding_indices"]
    rec = tokenizer.decode_tokens(tokens).clamp(0.0, 1.0)

print(f"{model_type} output: {tokens}")

plt.subplots(1, 2, figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(rec[0].permute(1, 2, 0).detach().cpu().numpy())
plt.title("Reconstructed Image")
plt.axis("off")





plt.show()