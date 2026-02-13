import os
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def simulate_zoom_crop(image, zoom_ratio, output_size):
    """
    Simulates a camera zoom by cropping the center of the image and resizing it.
    
    Args:
        image (PIL.Image): The source image.
        zoom_ratio (float): The zoom factor (e.g., 1.0, 2.0).
        output_size (tuple): The desired output resolution (width, height).
        
    Returns:
        PIL.Image: The simulated zoomed image.
    """
    w, h = image.size
    new_w = int(w / zoom_ratio)
    new_h = int(h / zoom_ratio)
    
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = (w + new_w) // 2
    bottom = (h + new_h) // 2
    
    cropped = image.crop((left, top, right, bottom))
    zoomed = cropped.resize(output_size, Image.Resampling.LANCZOS)
    return zoomed


class InpaintZoomFuseDataset(Dataset):
    """
    Dataset class that handles loading images, masks, and performing
    zoom simulation and fusion preprocessing.
    """
    def __init__(self, result_dir, data_dir, zoom_ratio=1.0, resolution=1024):
        self.nx_inpaint_paths = sorted(glob.glob(os.path.join(result_dir, '*')))
        self.x1_image_dir = os.path.join(data_dir, 'imgs')
        self.x1_mask_dir = os.path.join(data_dir, 'masks')
        
        self.zoom_ratio = zoom_ratio
        self.resolution = resolution

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.nx_inpaint_paths)

    def __getitem__(self, idx):
        nx_inpaint_path = self.nx_inpaint_paths[idx]
        fname = os.path.basename(nx_inpaint_path)

        x1_image_path = os.path.join(self.x1_image_dir, fname)
        x1_mask_path = os.path.join(self.x1_mask_dir, fname)

        # Load images
        nx_inpaint_img = Image.open(nx_inpaint_path).convert('RGB').resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        x1_image = Image.open(x1_image_path).convert('RGB').resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        x1_mask = Image.open(x1_mask_path).convert('L').resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)

        # Simulate zoom on original image and mask
        zoomed_image = simulate_zoom_crop(x1_image, self.zoom_ratio, output_size=nx_inpaint_img.size)
        zoomed_mask  = simulate_zoom_crop(x1_mask,  self.zoom_ratio, output_size=nx_inpaint_img.size)

        # Convert to numpy for blending
        nx_np = np.array(nx_inpaint_img).astype(np.float32)
        zoomed_img_np = np.array(zoomed_image).astype(np.float32)
        
        # Normalize mask to [0, 1]
        mask_np = np.array(zoomed_mask).astype(np.float32) / 255.0
        mask_np = np.expand_dims(mask_np, axis=-1)

        # Fuse: Blend the inpainted result with the original zoomed background using the mask
        fuse_np = (mask_np * nx_np) + ((1 - mask_np) * zoomed_img_np)
        fuse_np = fuse_np.clip(0, 255).astype(np.uint8)
        
        fuse_img = Image.fromarray(fuse_np)
        fuse_tensor = self.transform(fuse_img)

        return fuse_tensor, fname 


def save_tensor_image(tensor, save_path, resize_to=None):
    """
    Helper function to denormalize and save a tensor as an image.
    """
    tensor = tensor.detach().cpu().clone()
    tensor = tensor.clamp(-1, 1)  
    tensor = (tensor + 1) / 2.0      

    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Warning: Tensor contains NaN or Inf. Replacing with 0.")
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    image = transforms.ToPILImage()(tensor)
    if resize_to:
        image = image.resize(resize_to, Image.Resampling.LANCZOS)
    image.save(save_path)


def test_vae_batch(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load VAE model
    print(f"Loading VAE from: {args.trained_decoder_path}")
    vae = AutoencoderKL.from_pretrained(args.trained_decoder_path).to(device)
    vae.eval()

    # Initialize Dataset and DataLoader
    dataset = InpaintZoomFuseDataset(
        result_dir=args.result_dir,
        data_dir=args.data_dir,
        zoom_ratio=args.zoom_ratio,
        resolution=args.resolution
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Create output directories
    input_save_dir = os.path.join(args.output_dir, 'input')
    recon_save_dir = os.path.join(args.output_dir, 'reconstruct')
    os.makedirs(input_save_dir, exist_ok=True)
    os.makedirs(recon_save_dir, exist_ok=True)

    print(f"Starting inference on {len(dataset)} images...")

    for fuse_tensor, filenames in tqdm(dataloader, desc="Testing VAE"):
        
        fuse_tensor = fuse_tensor.to(device)
        fname = filenames[0] # Batch size is 1

        # Save Input Image (Fused)
        input_save_path = os.path.join(input_save_dir, fname)
        save_tensor_image(fuse_tensor, input_save_path, resize_to=(960, 540))

        # Perform VAE Reconstruction
        with torch.no_grad():
            posterior = vae.encode(fuse_tensor).latent_dist
            latents = posterior.sample()
            recon = vae.decode(latents, return_dict=False)[0]

        # Save Reconstructed Image
        recon_save_path = os.path.join(recon_save_dir, fname)
        save_tensor_image(recon, recon_save_path, resize_to=(960, 540))  

    print(f"Processing complete. Results saved to: {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="VAE Batch Inference Script")
    
    parser.add_argument("--trained_decoder_path", type=str, required=True,
                        help="Path to the trained VAE decoder checkpoint.")
    
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory containing the input images (e.g., inpainted results).")
    
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of the dataset containing 'imgs' and 'masks' subfolders.")
    
    parser.add_argument("--output_dir", type=str, default="output/vae_results",
                        help="Directory where output images will be saved.")

    parser.add_argument("--resolution", type=int, default=1024,
                        help="Image resolution for processing.")
    
    parser.add_argument("--zoom_ratio", type=float, default=1.0,
                        help="Zoom ratio for simulation.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Optional: Check if paths exist
    if not os.path.exists(args.trained_decoder_path):
        raise FileNotFoundError(f"Model path not found: {args.trained_decoder_path}")
    if not os.path.exists(args.result_dir):
        raise FileNotFoundError(f"Result directory not found: {args.result_dir}")

    test_vae_batch(args)
    
    
# python test_vae.py \
#   --trained_decoder_path /path/to/vae_checkpoint \
#   --result_dir /path/to/inpainted_images \
#   --data_dir /path/to/dataset_root \
#   --output_dir /path/to/save_outputs \
#   --zoom_ratio 3.0 \
#   --resolution 1024