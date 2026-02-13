# ReFocusEraser

This is the code for paper: "ReFocusEraser: Refocusing for Small Object Removal with Robust Context-Shadow Repair".

<p align = "center">
<img  src="src/Frameworkv6.png" width="800" />
</p>

## ⭐ Update
- [2026.02] Release the inference code.
- [2026.02] This repo is created.

### ✅ TODO
- [ ] Release the pretrained weights
- [✅] ~~Release the inference code~~


## Inference
Run the following command to try our model:
```shell
python inference.py \
  --img_path /path/to/imgs \
  --mask_path /path/to/masks \
  --output_path /path/to/output\
  --flux_path /path/to/FLUX.1-dev \
  --lora_path /path/to/ReFocusEraser_Checkpoint
```
