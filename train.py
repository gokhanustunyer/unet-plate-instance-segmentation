# -*- coding: utf-8 -*-
"""plate-segmentation-unet.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SAJqf8eeuW9YZB3LdZqazUivJUkKAq1s
"""

import os, shutil, subprocess
from google.colab import drive
drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/datasets/plate')

!pip install segmentation-models-pytorch tqdm

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from tqdm.auto import tqdm

with open("config.json", 'r', encoding='utf-8') as file:
    PLATE_DETECTOR = json.load(file)

PLATE_DETECTOR['COMPUTE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
PLATE_DETECTOR['DIMENSIONS'] = (128, 128)
print(f"Running on: {PLATE_DETECTOR['COMPUTE']}")

class LicensePlateDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None, class_count=3):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.class_count = class_count

        # Find all image files
        self.image_list = sorted([f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))])

        # Filter out images without corresponding masks
        valid_images = []
        for image_name in self.image_list:
            # Look for mask with various possible extensions
            mask_exists = False
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                image_stem = os.path.splitext(image_name)[0]
                mask_file = os.path.join(self.masks_path, image_stem + ext)
                if os.path.exists(mask_file):
                    mask_exists = True
                    break

            if mask_exists:
                valid_images.append(image_name)
            else:
                print(f"Warning: No mask found for {image_name}")

        self.image_list = valid_images

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_file = os.path.join(self.images_path, image_name)

        # Find the corresponding mask file
        image_stem = os.path.splitext(image_name)[0]
        mask_file = None
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            potential_mask = os.path.join(self.masks_path, image_stem + ext)
            if os.path.exists(potential_mask):
                mask_file = potential_mask
                break

        if mask_file is None:
            mask_file = os.path.join(self.masks_path, image_name)  # Default fallback

        # Load image
        img = cv2.imread(image_file)
        if img is None:
            raise ValueError(f"Failed to load image: {image_file}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_file}")

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        # Process mask values
        if isinstance(mask, np.ndarray):
            # Ensure mask values match expected class range
            if mask.max() > self.class_count - 1:
                # Create mapping for mask values
                distinct_values = np.unique(mask)
                value_to_class = {val: idx for idx, val in enumerate(distinct_values)}
                for old_val, new_val in value_to_class.items():
                    mask[mask == old_val] = new_val

            # Convert to appropriate tensor type
            mask = torch.from_numpy(mask).long()
        elif isinstance(mask, torch.Tensor) and mask.dtype != torch.int64:
            mask = mask.long()

        return img, mask

def get_train_augmentation():
    return A.Compose([
        A.Resize(height=PLATE_DETECTOR['DIMENSIONS'][0], width=PLATE_DETECTOR['DIMENSIONS'][1]),
        A.HorizontalFlip(p=0.5),  # Added data augmentation
        A.RandomBrightnessContrast(p=0.2),  # Added data augmentation
        ToTensorV2(),
    ])

def get_valid_augmentation():
    return A.Compose([
        A.Resize(height=PLATE_DETECTOR['DIMENSIONS'][0], width=PLATE_DETECTOR['DIMENSIONS'][1]),
        ToTensorV2(),
    ])

# Initialize datasets
train_dataset = LicensePlateDataset(
    images_path=os.path.join(PLATE_DETECTOR['ROOT_DIR'], 'train/images'),
    masks_path=os.path.join(PLATE_DETECTOR['ROOT_DIR'], 'train/masks'),
    transform=get_train_augmentation(),
    class_count=PLATE_DETECTOR['SEGMENT_CLASSES']
)
valid_dataset = LicensePlateDataset(
    images_path=os.path.join(PLATE_DETECTOR['ROOT_DIR'], 'valid/images'),
    masks_path=os.path.join(PLATE_DETECTOR['ROOT_DIR'], 'valid/masks'),
    transform=get_valid_augmentation(),
    class_count=PLATE_DETECTOR['SEGMENT_CLASSES']
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=PLATE_DETECTOR['SAMPLES_PER_BATCH'],
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=PLATE_DETECTOR['SAMPLES_PER_BATCH'],
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# Create model
plate_model = smp.Unet(
    encoder_name=PLATE_DETECTOR['BASE_MODEL'],
    encoder_weights=PLATE_DETECTOR['TRANSFER_LEARNING'],
    classes=PLATE_DETECTOR['SEGMENT_CLASSES'],
    activation=None,  # No activation for multiclass segmentation
)

plate_model.to(PLATE_DETECTOR['COMPUTE'])

# Define loss function and optimizer
loss_function = DiceLoss('multiclass', classes=PLATE_DETECTOR['SEGMENT_CLASSES'])
optimizer = torch.optim.Adam(plate_model.parameters(), lr=PLATE_DETECTOR['LEARNING_SPEED'])
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.3, verbose=True)

def run_training_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    # Create a TQDM progress bar for the batches
    batch_progress = tqdm(dataloader, desc="Training", leave=False, position=1)

    for images, masks in batch_progress:
        # Move data to device
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.long)

        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update total loss and progress bar
        current_loss = loss.item()
        total_loss += current_loss
        batch_progress.set_postfix({'batch_loss': f'{current_loss:.4f}'})

    return total_loss / len(dataloader)

def run_validation_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    # Handle empty dataloader
    if len(dataloader) == 0:
        print("WARNING: Empty validation dataloader - skipping validation step")
        return 0.0

    # Create a TQDM progress bar for validation batches
    batch_progress = tqdm(dataloader, desc="Validation", leave=False, position=1)

    with torch.no_grad():
        for images, masks in batch_progress:
            # Move data to device
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)

            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, masks)

            # Update total loss and progress bar
            current_loss = loss.item()
            total_loss += current_loss
            batch_progress.set_postfix({'val_loss': f'{current_loss:.4f}'})

    return total_loss / len(dataloader)

def detect_license_plates(model, image, device, threshold=0.5):
    # Create a copy of input image
    img_copy = image.copy()

    # Resize for model input
    img_resized = cv2.resize(img_copy, (PLATE_DETECTOR['DIMENSIONS'][1], PLATE_DETECTOR['DIMENSIONS'][0]))

    # Process image for model
    transform = get_valid_augmentation()
    transformed = transform(image=img_resized)
    model_input = transformed['image'].unsqueeze(0).to(device)

    # Ensure correct data type
    model_input = model_input.float()

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(model_input)
        # Get predicted class for each pixel
        class_map = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Create instance segmentation
    instances = np.zeros_like(class_map)
    instance_id = 1

    # Find connected components for each class
    for class_id in range(PLATE_DETECTOR['SEGMENT_CLASSES']):
        # Create binary mask for current class
        class_binary = (class_map == class_id).astype(np.uint8)
        if np.sum(class_binary) > 0:
            num_components, components = cv2.connectedComponents(class_binary)

            # Label each component with unique ID
            for comp_id in range(1, num_components):
                instance_id += 1

    return instances, class_map

def train_plate_detector(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, device):
    best_val_loss = float('inf')
    metrics = {'train_loss': [], 'valid_loss': []}

    # Check for empty dataloaders
    if len(train_loader) == 0:
        print("ERROR: Training dataloader is empty! Cannot train the model.")
        return metrics

    print(f"Training using {len(train_loader)} batches per epoch")
    print(f"Validating using {len(valid_loader)} batches per epoch")

    # Create a TQDM progress bar for epochs
    epoch_progress = tqdm(range(epochs), desc="Training Progress", position=0)

    for epoch in epoch_progress:
        # Train with progress bar
        train_loss = run_training_epoch(model, train_loader, optimizer, criterion, device)

        # Validate with progress bar if valid_loader is not empty
        if len(valid_loader) > 0:
            valid_loss = run_validation_epoch(model, valid_loader, criterion, device)
        else:
            print("WARNING: Skipping validation (empty dataloader)")
            valid_loss = train_loss  # Fallback

        # Update learning rate
        scheduler.step(valid_loss)

        # Record metrics
        metrics['train_loss'].append(train_loss)
        metrics['valid_loss'].append(valid_loss)

        # Update the progress bar description with loss values
        epoch_progress.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'valid_loss': f'{valid_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

        # Save best model
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), 'best_plate_detector.pth')
            epoch_progress.write(f'Model saved with loss: {best_val_loss:.4f}')

    return metrics

def visualize_detections(model, test_dir, device, num_samples=5, save_output=True):
    test_images = sorted(os.listdir(os.path.join(test_dir, "images")))
    transform = get_valid_augmentation()

    # Create colormap for visualization
    cmap = plt.cm.get_cmap('tab10', PLATE_DETECTOR['SEGMENT_CLASSES'])

    # Create output directory
    results_dir = os.path.join(os.path.dirname(test_dir), 'plate_results')
    if save_output and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples*5))

    for i in range(min(num_samples, len(test_images))):
        # Load image
        img_name = test_images[i]
        img_path = os.path.join(test_dir, "images", img_name)
        print(f"Processing license plate image: {img_path}")
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Prepare for model
        transformed = transform(image=image)
        model_input = transformed['image'].to(device, dtype=torch.float32).unsqueeze(0)

        # Get prediction
        model.eval()
        with torch.no_grad():
            prediction = model(model_input)
            # Get class predictions
            class_prediction = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()

        # Display images
        ax1 = axes[i, 0] if num_samples > 1 else axes[0]
        ax2 = axes[i, 1] if num_samples > 1 else axes[1]
        ax3 = axes[i, 2] if num_samples > 1 else axes[2]

        ax1.imshow(image)
        ax1.set_title(f'Original License Plate Image')
        ax1.axis('off')

        # Display class prediction
        im = ax2.imshow(class_prediction, cmap=cmap, vmin=0, vmax=PLATE_DETECTOR['SEGMENT_CLASSES']-1)
        ax2.set_title(f'Segmentation Classes')
        ax2.axis('off')

        # Create instance segmentation
        instances = np.zeros_like(class_prediction)
        instance_id = 1

        # Find connected components for each class
        for class_id in range(PLATE_DETECTOR['SEGMENT_CLASSES']):
            # Create binary mask for current class
            class_binary = (class_prediction == class_id).astype(np.uint8)
            if np.sum(class_binary) > 0:
                num_components, components = cv2.connectedComponents(class_binary)

                # Label each component with unique ID
                for comp_id in range(1, num_components):
                    instances[components == comp_id] = instance_id
                    instance_id += 1

        # Display instance segmentation
        instance_display = np.zeros((*instances.shape, 3), dtype=np.uint8)

        # Colorize instances
        for j in range(1, int(instances.max()) + 1):
            color = np.random.randint(0, 255, size=3)
            instance_display[instances == j] = color

        ax3.imshow(instance_display)
        ax3.set_title(f'License Plate Instances ({int(instances.max())} instances)')
        ax3.axis('off')

        # Save results
        if save_output:
            # Get base name without extension
            base_name = os.path.splitext(img_name)[0]

            # Save original image
            original_path = os.path.join(results_dir, f"{base_name}_original.png")
            plt.imsave(original_path, image)

            # Save class prediction
            pred_path = os.path.join(results_dir, f"{base_name}_segmentation.png")
            plt.imsave(pred_path, class_prediction, cmap=cmap)

            # Save instance visualization
            instance_path = os.path.join(results_dir, f"{base_name}_instances.png")
            cv2.imwrite(instance_path, cv2.cvtColor(instance_display, cv2.COLOR_RGB2BGR))

    plt.tight_layout()

    if save_output:
        fig_path = os.path.join(results_dir, "plate_detection_results.png")
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)

    plt.show()

    if save_output:
        print(f"Detection results saved to {results_dir}")

# Train model
print("Starting license plate detector training...")
metrics = train_plate_detector(
    model=plate_model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=loss_function,
    optimizer=optimizer,
    scheduler=lr_scheduler,
    epochs=PLATE_DETECTOR['TRAINING_ROUNDS'],
    device=PLATE_DETECTOR['COMPUTE']
)

# Load best model
plate_model.load_state_dict(torch.load('best_plate_detector.pth'))

# Visualize test predictions
test_dir = os.path.join(PLATE_DETECTOR['ROOT_DIR'], 'valid')
print("Visualizing license plate detections...", test_dir)
visualize_detections(plate_model, test_dir, PLATE_DETECTOR['COMPUTE'])