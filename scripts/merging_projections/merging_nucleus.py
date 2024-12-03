import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2 as cv
import re
from pathlib import Path

input_folder = ''
output_folder = ''

def extract_z_info(filename):
    """Extract position and Z-stack information from filename."""
    # Match pattern like "Position 1 - 1_Z00_C0" or "Position 1_Z01_C0"
    match = re.search(r'Position (\d+(?:\s*-\s*\d+)?)[_ ](Z\d+)_(C\d+)', filename)
    
    if match:
        position = match.group(1).replace(' ', '')  # Remove spaces from position
        z_number = int(match.group(2)[1:])  # Extract number from Z00, Z01, etc.
        channel = match.group(3)
        return position, z_number, channel
    else:
        return None, None, None

def flatten_z_stacks(input_folder, output_folder):
    """
    Create flattened 2D images by combining all Z-projections for each position.
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all .tif files
    all_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
    print(f"\nTotal .tif files found: {len(all_files)}")
    
    if not all_files:
        raise ValueError(f"No .tif files found in {input_folder}")
    
    # Group files by position and channel
    position_stacks = {}
    for filename in all_files:
        position, z_number, channel = extract_z_info(filename)
        if position is not None:
            key = (position, channel)
            if key not in position_stacks:
                position_stacks[key] = []
            position_stacks[key].append((z_number, filename))
    
    # Print found positions and channels
    print(f"\nFound {len(position_stacks)} position-channel combinations:")
    for (pos, channel), files in position_stacks.items():
        print(f"Position {pos}, {channel}: {len(files)} Z-stacks")
    
    # Process each position-channel combination
    for (position, channel), files in position_stacks.items():
        print(f"\nProcessing Position {position}, {channel}")
        
        # Sort files by Z number
        files.sort()  # Will sort based on Z-number
        
        # Initialize variables for stack processing
        stack = []
        first_image = True
        
        # Process each Z-stack
        for z_number, filename in files:
            img_path = os.path.join(input_folder, filename)
            print(f"  Reading Z-stack {z_number}: {filename}")
            
            if os.path.exists(img_path):
                # Read image in original bit depth
                image = cv.imread(img_path, cv.IMREAD_UNCHANGED)
                
                if image is not None:
                    if first_image:
                        # Initialize image properties from first image
                        img_height, img_width = image.shape[:2]
                        first_image = False
                        
                    # Convert to float32 for processing
                    image = image.astype(np.float32)
                    
                    # Add to stack
                    stack.append(image)
                    print(f"    Successfully added to stack")
                else:
                    print(f"    Warning: Could not read image")
            else:
                print(f"    Warning: File not found")
        
        if stack:
            # Convert stack to numpy array
            stack = np.array(stack)
            print(f"  Processing stack of shape: {stack.shape}")
            
            # Create maximum intensity projection
            flattened = np.max(stack, axis=0)
            
            # Normalize the image
            flattened_min = np.min(flattened)
            flattened_max = np.max(flattened)
            if flattened_max > flattened_min:
                flattened = ((flattened - flattened_min) * 
                           (255.0 / (flattened_max - flattened_min)))
            
            # Convert to uint8
            flattened = flattened.astype(np.uint8)
            
            # Apply CLAHE for contrast enhancement
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(flattened)
            
            # Save both versions
            base_output = f"flattened_position_{position}_{channel}"
            
            # Save original flattened image
            output_path = os.path.join(output_folder, f"{base_output}.tif")
            cv.imwrite(output_path, flattened)
            print(f"Saved flattened image: {output_path}")
            
            # Save enhanced version
            enhanced_path = os.path.join(output_folder, f"{base_output}_enhanced.tif")
            cv.imwrite(enhanced_path, enhanced)
            print(f"Saved enhanced version: {enhanced_path}")
            
            # Create and save a color-coded depth map
            z_indices = np.argmax(stack, axis=0)
            depth_map = ((z_indices.astype(float) / (len(stack) - 1)) * 255).astype(np.uint8)
            depth_map_colored = cv.applyColorMap(depth_map, cv.COLORMAP_JET)
            depth_path = os.path.join(output_folder, f"{base_output}_depth_map.tif")
            cv.imwrite(depth_path, depth_map_colored)
            print(f"Saved depth map: {depth_path}")
            
        else:
            print(f"Warning: No valid images found for Position {position}, {channel}")

# Example usage
if __name__ == "__main__":
    input_folder = input_folder  # Replace with your input folder path
    output_folder = output_folder  # Replace with your output folder path
    
    try:
        flatten_z_stacks(input_folder, output_folder)
    except Exception as e:
        print(f"Error: {str(e)}")