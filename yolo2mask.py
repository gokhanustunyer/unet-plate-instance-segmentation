import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_mask_from_yolo_polygon(label_path, img_shape, output_path):
    """
    Convert YOLO polygon format to a mask image
    
    Args:
        label_path (str): Path to the YOLO label file
        img_shape (tuple): Shape of the original image (height, width)
        output_path (str): Path to save the mask image
    """
    height, width = img_shape
    # Create an empty mask with same dimensions as the image
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Read label file
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5:  # Skip invalid lines
                continue
                
            # Class ID is the first number
            class_id = int(parts[0]) + 1  # Adding 1 because 0 will be background
            
            # Extract points (x,y pairs) for polygon
            points = []
            for i in range(1, len(parts), 2):
                if i+1 < len(parts):
                    # Convert normalized coordinates to pixel coordinates
                    x = float(parts[i]) * width
                    y = float(parts[i+1]) * height
                    points.append([x, y])
            
            # Convert points to the format expected by fillPoly
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Fill the polygon with the class ID
            cv2.fillPoly(mask, [pts], class_id)
    
    # Save the mask image
    cv2.imwrite(output_path, mask)
    return mask

def process_dataset(base_dir, output_base_dir):
    """
    Process the entire dataset (train, test, valid)
    
    Args:
        base_dir (str): Base directory containing train, test, valid folders
        output_base_dir (str): Base directory to save masks
    """
    # Create output directories if they don't exist
    for split in ['train', 'test', 'valid']:
        output_dir = os.path.join(output_base_dir, split, 'masks')
        os.makedirs(output_dir, exist_ok=True)
        print(output_dir)
        
        # Get all image paths
        img_dir = os.path.join(base_dir, split, 'images')
        img_paths = glob(os.path.join(img_dir, '*.jpg')) + glob(os.path.join(img_dir, '*.png'))
        
        for img_path in tqdm(img_paths, desc=f"Processing {split}"):
            # Get image name and corresponding label path
            img_name = os.path.basename(img_path)
            img_name_without_ext = os.path.splitext(img_name)[0]
            label_path = os.path.join(base_dir, split, 'labels', f"{img_name_without_ext}.txt")
            
            # Skip if label doesn't exist
            if not os.path.exists(label_path):
                print(f"Warning: No label file for {img_path}")
                continue
                
            # Read image to get its shape
            img = cv2.imread(img_path)
            img_shape = img.shape[:2]  # (height, width)
            
            # Create and save mask
            output_path = os.path.join(output_base_dir, split, 'masks', f"{img_name_without_ext}.png")
            create_mask_from_yolo_polygon(label_path, img_shape, output_path)

def visualize_sample(img_path, mask_path):
    """
    Visualize a sample image and its mask for verification
    
    Args:
        img_path (str): Path to the original image
        mask_path (str): Path to the mask image
    """
    # Read images
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Display
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='tab20')
    plt.title('Mask')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Configuration
    input_base_dir = "./plate"  # Base directory containing train, test, valid folders
    output_base_dir = "./plate"  # Base directory to save masks
    
    # Process dataset
    process_dataset(input_base_dir, output_base_dir)
    
    # Optional: Visualize a sample to verify conversion
    # Uncomment and modify paths as needed
    # visualize_sample("train/images/sample.jpg", "train/masks/sample.png")
    
    print("Conversion completed successfully!")