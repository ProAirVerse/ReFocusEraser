import torch
import os
import argparse
from diffusers.utils import load_image, check_min_version
from models.camctrl_transformer import CamCtrlFluxTransformer2DModel
from pipelines.pipeline_camtrl_removal import FluxControlSingleScaleRemovalPipeline
from geocalib import GeoCalib
from PIL import Image

# Check diffusers version compatibility
check_min_version("0.30.2")

def parse_args():
    parser = argparse.ArgumentParser(description="OmniEraser Single Image Inference")
    
    # Input/Output paths
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask image")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the result")
    
    # Model paths (Users need to change these or pass them via command line)
    parser.add_argument("--flux_path", type=str, 
                        default="/path/to/FLUX.1-dev",
                        help="Path to the local FLUX.1-dev model snapshot")
    parser.add_argument("--lora_path", type=str, 
                        default="theSure/Omnieraser",
                        help="Path to the LoRA weights directory or HuggingFace repo id")
    parser.add_argument("--lora_filename", type=str, 
                        default="pytorch_lora_weights.safetensors",
                        help="Filename of the LoRA weights")
    
    # Inference parameters
    parser.add_argument("--prompt", type=str, default="There is nothing here.", help="Text prompt for inpainting")
    parser.add_argument("--process_size", type=int, nargs=2, default=[1024, 1024], help="Resolution for model inference (W H)")
    parser.add_argument("--output_size", type=int, nargs=2, default=[960, 540], help="Final output resolution (W H)")
    parser.add_argument("--zoom_ratio", type=str, default="1x", help="Zoom ratio parameter")
    parser.add_argument("--seed", type=int, default=24, help="Random seed for reproducibility")
    
    return parser.parse_args()

def build_pipeline(args):
    """
    Loads the Transformer, applies channel expansion (monkey patching), 
    and initializes the custom Flux pipeline.
    """
    print(f"Loading Flux Transformer from: {args.flux_path}")
    
    # 1. Load the Transformer model
    transformer = CamCtrlFluxTransformer2DModel.from_pretrained(
        args.flux_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    # 2. Expand input channels (Monkey Patching)
    # The model needs to accept additional control inputs (image + mask), 
    # so we expand the input embedding layer.
    with torch.no_grad(): 
        initial_input_channels = transformer.config.in_channels
        
        # Create a new linear layer with expanded input features (x4)
        new_linear = torch.nn.Linear(
            transformer.x_embedder.in_features * 4,
            transformer.x_embedder.out_features,
            bias=transformer.x_embedder.bias is not None,
            dtype=transformer.dtype,
            device=transformer.device,
        )
        
        # Initialize new weights
        new_linear.weight.zero_()
        # Copy original weights to the first chunk of the new layer
        new_linear.weight[:, :initial_input_channels].copy_(transformer.x_embedder.weight)
        
        if transformer.x_embedder.bias is not None:
            new_linear.bias.copy_(transformer.x_embedder.bias)
            
        # Replace the embedding layer in the transformer
        transformer.x_embedder = new_linear
        transformer.register_to_config(in_channels=initial_input_channels * 4)

    # 3. Initialize the custom Pipeline
    print("Initializing FluxControlSingleScaleRemovalPipeline...")
    pipe = FluxControlSingleScaleRemovalPipeline.from_pretrained(
        args.flux_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    pipe.transformer.to(torch.bfloat16)

    # Verify channel consistency
    assert (
        pipe.transformer.config.in_channels == initial_input_channels * 4
    ), "Transformer input channels mismatch after patching."

    # 4. Load LoRA weights
    print(f"Loading LoRA from: {args.lora_path}")
    pipe.load_lora_weights(args.lora_path, weight_name=args.lora_filename)

    # 5. Initialize Camera Model (GeoCalib)
    print("Loading GeoCalib model...")
    camera_model = GeoCalib(weights="pinhole").to("cuda")

    return pipe, camera_model

def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)

    # Initialize models
    pipe, camera_model = build_pipeline(args)

    # Check input files
    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"Image not found at: {args.img_path}")
    if not os.path.exists(args.mask_path):
        raise FileNotFoundError(f"Mask not found at: {args.mask_path}")

    print(f"Processing image: {args.img_path}")

    # Load and preprocess images
    # Note: Resize inputs to the processing resolution (default 1024x1024)
    image = load_image(args.img_path).convert("RGB").resize(tuple(args.process_size))
    mask = load_image(args.mask_path).convert("RGB").resize(tuple(args.process_size))
    
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # Run inference
    result = pipe(
        prompt=args.prompt,
        control_image=image,
        control_mask=mask,
        num_inference_steps=28,
        guidance_scale=3.5,
        generator=generator,
        max_sequence_length=512,
        height=args.process_size[1],
        width=args.process_size[0],
        camera_model=camera_model,
        test_zomm_ratio=args.zoom_ratio,
    ).images[0]

    # Post-processing: Resize to target output resolution
    # Using LANCZOS resampling for high-quality downscaling/upscaling
    result = result.resize(tuple(args.output_size), Image.Resampling.LANCZOS)
    
    # Save result
    result.save(args.output_path)
    print(f"Successfully saved result to: {args.output_path}")

if __name__ == "__main__":
    main()
    
    