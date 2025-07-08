import json
import os
import time
import sys
import gc

import logging

import numpy as np

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
import torchvision.utils
from torch.utils.data import TensorDataset, Subset, ConcatDataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

print("Starting CCMNIST preprocessing script...")

root = '/home/chenz1/toorange/Data/colorized-MNIST-master/'

# LOCAL_IMAGE_LIST_PATH = 'metas/intra_test/train_label.json'

DEST = '/home/chenz1/toorange/Data/CCMNIST1/'

print(f"Using root directory: {root}")
print(f"Output destination: {DEST}")

# Create output directories if they don't exist
for i in range(3):
    os.makedirs(os.path.join(DEST, str(i)), exist_ok=True)
    for label in range(2):  # Binary labels 0 and 1
        os.makedirs(os.path.join(DEST, str(i), str(label)), exist_ok=True)

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

print("Loading MNIST datasets...")
original_dataset_tr = MNIST(root, train=True, download=True, transform=transform)
original_dataset_te = MNIST(root, train=False, download=True, transform=transform)
print(f"Loaded training dataset with {len(original_dataset_tr)} samples")
print(f"Loaded test dataset with {len(original_dataset_te)} samples")

print("Concatenating datasets...")
data = ConcatDataset([original_dataset_tr, original_dataset_te])
print(f"Combined dataset size: {len(data)}")

print("Extracting images and labels...")
original_images = torch.cat([img for img, _ in data])
original_labels = torch.cat([torch.tensor(label).unsqueeze(0) for _, label in data])
print(f"Extracted {len(original_images)} images with shape {original_images.shape}")
print(f"Extracted {len(original_labels)} labels")

print("Shuffling data...")
shuffle = torch.randperm(len(original_images))
original_images = original_images[shuffle]
original_labels = original_labels[shuffle]
print("Data shuffled successfully")

# Clear memory
del original_dataset_tr, original_dataset_te, data
gc.collect()

datasets = []
dict = {}
num = 1
environments = [0,1,2]

# Set to False to skip saving individual images (saves memory and time)
SAVE_INDIVIDUAL_IMAGES = True

print(f"Will process {len(environments)} environments: {environments}")
print(f"Save individual images: {SAVE_INDIVIDUAL_IMAGES}")


def torch_bernoulli_(p, size):
    return (torch.rand(size) < p).float()


def torch_xor_(a, b):
    return (a - b).abs()

def colored_dataset(images, labels, env):
    print(f"Processing environment {env} with {len(images)} images...")

    x = torch.zeros(len(images), 1, 224, 224)

    environement_color = -1
    if env == 0:
        environement_color = 0.1
    elif env == 1:
        environement_color = 0.3
    elif env == 2:
        environement_color = 0.5

    print(f"Environment {env} color probability: {environement_color}")

    # Assign a binary label based on the digit
    fake_labels = (labels < 5).float()
    print(f"Created binary labels: {fake_labels.sum().item()} ones, {len(fake_labels) - fake_labels.sum().item()} zeros")

    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor_(fake_labels, torch_bernoulli_(environement_color, len(fake_labels)))
    print(f"Assigned colors: {colors.sum().item()} colored, {len(colors) - colors.sum().item()} background")

    x = torch.squeeze(images, dim=1)
    ones = torch.ones_like(x)
    x = torch.logical_xor(x, ones).float()

    zeros = torch.zeros_like(x)
    bg1 = torch.logical_xor(x, zeros).float()
    bg2 = bg1
    ones = torch.ones_like(bg2)
    x = torch.logical_xor(bg1,ones)

    print("Applying color transformations...")
    for i in range(x.size(0)):
        if colors[i] == 1.0:
            x[i,:,:][x[i,:,:] == 0.0] = 1.0
        else:
            bg2[i,:,:] = 0.0

    print(f"Creating RGB channels for environment {env}...")
    if env == 0:
        x = torch.stack([x, bg2, bg2], dim=1)
    elif env == 1:
        x = torch.stack([bg2, x, bg2], dim=1)
    elif env == 2:
        x = torch.stack([bg2, bg2, x], dim=1)

    # Apply the color to the image by zeroing out the other color channel
    # x[torch.tensor(range(len(x))), (1 - colors).long(), :, :] *= 0
    # x[torch.tensor(range(len(x))), :, :, :][x[torch.tensor(range(len(x))), :, :, :] == 0] = 1

    x = x.float()  # .div_(255.0)
    y = fake_labels.view(-1).long()

    print(f"Environment {env} processing complete. Output shape: {x.shape}")
    return [x,y,colors]

print("Starting dataset creation for each environment...")
for i in range(len(environments)):
    print(f"\nProcessing environment {environments[i]} ({i+1}/{len(environments)})...")
    images = original_images[i::len(environments)]
    labels = original_labels[i::len(environments)]
    print(f"Selected {len(images)} images for environment {environments[i]}")
    datasets.append(colored_dataset(images, labels, environments[i]))

print("\nSaving processed datasets...")
for i in range(len(environments)):
    print(f"Saving environment {environments[i]}...")
    outpath = DEST+str(i)+'/'
    x = datasets[i][0]
    y = datasets[i][1]
    z = datasets[i][2]

    print(f"Environment {environments[i]} - Images: {x.shape}, Labels: {y.shape}, Colors: {z.shape}")

    save_path = "/data/CCMNIST1/" + str(i) + "_224.pt"
    print(f"Saving to: {save_path}")
    torch.save((x, y, z), save_path)
    print(f"Environment {environments[i]} saved successfully")

    # Save individual images in batches to avoid memory issues
    if SAVE_INDIVIDUAL_IMAGES:
        print(f"Saving individual images for environment {environments[i]}...")
        batch_size = 50  # Reduced batch size to prevent memory issues
        total_images = y.size(0)

        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            print(f"Processing batch {batch_start//batch_size + 1}/{(total_images + batch_size - 1)//batch_size} (images {batch_start}-{batch_end-1})")

            for j in range(batch_start, batch_end):
                try:
                    image_tensor = x[j].clone()  # Clone to avoid memory issues
                    label = y[j].item()
                    color = z[j].item()
                    img_name = str(num)+'.png'
                    dict[img_name]=[i,label,color]

                    # Ensure the directory exists
                    save_dir = os.path.join(outpath, str(label))
                    os.makedirs(save_dir, exist_ok=True)

                    torchvision.utils.save_image(image_tensor, os.path.join(save_dir, img_name))
                    num += 1

                    # Clear the cloned tensor immediately
                    del image_tensor

                    # Clear memory more frequently
                    if j % 25 == 0:
                        gc.collect()

                except Exception as e:
                    print(f"Error saving image {j}: {e}")
                    continue

            # Clear memory after each batch
            gc.collect()
            print(f"Batch {batch_start//batch_size + 1} completed")
    else:
        print(f"Skipping individual image saving for environment {environments[i]}")

# Save metadata
if SAVE_INDIVIDUAL_IMAGES:
    print("Saving metadata...")
    with open(DEST+'data.json', 'w') as fp:
        json.dump(dict, fp)
    print("Metadata saved successfully")
else:
    print("Skipping metadata saving (no individual images saved)")

print("\nCCMNIST preprocessing completed successfully!")