import os
import glob
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

def simulate_zoom_crop(image, zoom_ratio, output_size):
    """
    Simulates a camera zoom by cropping the center of the image and resizing it.
    
    Args:
        image (PIL.Image): The source image (e.g., 1x zoom).
        zoom_ratio (float): The zoom factor (e.g., 2.0, 3.0).
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

def get_center_crop_coords(img_size, crop_ratio):
    """
    Calculates the coordinates for a center crop based on a ratio.
    
    Args:
        img_size (tuple): (width, height) of the original image.
        crop_ratio (float): Ratio of the crop size to the original size (0.0 to 1.0).
        
    Returns:
        tuple: (left, top, right, bottom) coordinates.
    """
    width, height = img_size
    target_width = width * crop_ratio
    target_height = height * crop_ratio
    
    left = int(np.ceil((width - target_width) / 2.0))
    top = int(np.ceil((height - target_height) / 2.0))
    right = int(np.floor((width + target_width) / 2.0))
    bottom = int(np.floor((height + target_height) / 2.0))
    
    return left, top, right, bottom

def paste_via_mask(result_img, x1_img, x1_mask, zoom_ratio):
    """
    Pastes the result image back onto the 1x image using a mask for seamless blending.
    
    This method first simulates the zoom on the 1x image and mask to match the 
    result image's perspective. It then blends the result with the simulated 
    zoom image using the mask, and finally pastes this blended patch back onto 
    the original 1x image.
    """
    # 1. Simulate the zoom view from the original 1x image and mask
    zoomed_img_ref = simulate_zoom_crop(x1_img, zoom_ratio, result_img.size)
    zoomed_mask_ref = simulate_zoom_crop(x1_mask, zoom_ratio, result_img.size).convert("L")
    
    # 2. Normalize mask to [0, 1]
    mask_np = np.array(zoomed_mask_ref).astype(np.float32) / 255.0
    mask_np = np.expand_dims(mask_np, axis=-1) # Expand for broadcasting
    
    # 3. Blend: (Mask * Result) + ((1 - Mask) * Original_Zoomed_View)
    result_np = np.array(result_img).astype(np.float32)
    zoomed_ref_np = np.array(zoomed_img_ref).astype(np.float32)
    
    fused_np = (mask_np * result_np) + ((1 - mask_np) * zoomed_ref_np)
    fused_np = fused_np.clip(0, 255).astype(np.uint8)
    fused_patch = Image.fromarray(fused_np)
    
    # 4. Paste the blended patch back into the original 1x image canvas
    final_canvas = x1_img.copy()
    crop_ratio = 1.0 / zoom_ratio
    left, top, right, bottom = get_center_crop_coords(x1_img.size, crop_ratio)
    
    # Resize the patch to fit the crop area in the 1x image
    fused_patch_resized = fused_patch.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    final_canvas.paste(fused_patch_resized, (left, top))
    
    return final_canvas

def paste_direct(result_img, x1_img, zoom_ratio):
    """
    Directly pastes the result image into the center of the 1x image.
    
    This method assumes the result image corresponds exactly to the center 
    crop of the 1x image at the given zoom ratio.
    """
    crop_ratio = 1.0 / zoom_ratio
    left, top, right, bottom = get_center_crop_coords(x1_img.size, crop_ratio)
    
    # Resize result to match the target area in the 1x image
    resized_result = result_img.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    
    final_canvas = x1_img.copy()
    final_canvas.paste(resized_result, (left, top))
    
    return final_canvas

def process_dataset(result_dir, data_dir, output_dir, zoom_ratio=3.0, method="direct"):
    """
    Batch processes images to paste zoomed results back onto original 1x frames.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get list of result images
    result_paths = sorted(glob.glob(os.path.join(result_dir, '*')))
    print(f"Found {len(result_paths)} images to process.")

    # Define subdirectories for 1x data
    x1_imgs_dir = os.path.join(data_dir, 'imgs')
    x1_masks_dir = os.path.join(data_dir, 'masks')

    for path in tqdm(result_paths, desc="Processing"):
        filename = os.path.basename(path)
        
        # Construct paths for corresponding 1x data
        x1_img_path = os.path.join(x1_imgs_dir, filename)
        x1_mask_path = os.path.join(x1_masks_dir, filename)

        # Validation
        if not os.path.exists(x1_img_path):
            print(f"[Warning] Skipping: 1x image not found for {filename}")
            continue

        # Load images
        try:
            result_img = Image.open(path).convert("RGB")
            x1_img = Image.open(x1_img_path).convert("RGB")
        except Exception as e:
            print(f"[Error] Could not load images for {filename}: {e}")
            continue

        # Process based on selected method
        if method == "mask":
            if not os.path.exists(x1_mask_path):
                print(f"[Warning] Skipping: Mask not found for {filename}")
                continue
            x1_mask = Image.open(x1_mask_path)
            final_output = paste_via_mask(result_img, x1_img, x1_mask, zoom_ratio)
            
        elif method == "direct":
            final_output = paste_direct(result_img, x1_img, zoom_ratio)
            
        else:
            raise ValueError(f"Invalid method: {method}. Choose 'mask' or 'direct'.")

        # Save result
        save_path = os.path.join(output_dir, filename)
        final_output.save(save_path)

    print(f"Processing complete. Results saved to: {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Paste zoomed result images back onto the original 1x canvas.")
    
    parser.add_argument("--result_dir", type=str, required=True, 
                        help="Directory containing the processed (zoomed) result images.")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Root directory of the 1x dataset (must contain 'imgs' and 'masks' subfolders).")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory where the final pasted images will be saved.")
    parser.add_argument("--zoom_ratio", type=float, default=3.0, 
                        help="The zoom factor used for the result images (e.g., 2.0, 3.0).")
    parser.add_argument("--method", type=str, choices=["direct", "mask"], default="direct", 
                        help="Pasting method: 'direct' (simple paste) or 'mask' (blend using mask).")

    return parser.parse_args()

if __name__ == "__main__":

    process_dataset(
        result_dir="/path/to/result",
        data_dir="/path/to/data",
        output_dir="/path/to/output",
        zoom_ratio=3.0,
        method="mask"
    )