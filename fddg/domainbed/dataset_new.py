import json
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision import transforms
import os
from pathlib import Path

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ObjectDetectionDataset(Dataset):
    def __init__(self, json_path, images_dir, transform=None, augment=False):
        """
        Args:
            json_path (string): Path to the JSON file with annotations
            images_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
            augment (bool): Whether to apply data augmentation
        """
        self.json_path = json_path
        self.images_dir = Path(images_dir)
        self.augment = augment

        # Define base transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Define augmentation transforms
        self.augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Use provided transform if specified, otherwise use base/augment transform
        self.transform = transform if transform is not None else (
            self.augment_transform if augment else self.base_transform
        )

        # Load annotations
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)

        # Create a mapping from image_name to list of annotations
        self.image_to_annotations = {}
        for ann in self.annotations:
            img_name = ann['image_name']
            if img_name not in self.image_to_annotations:
                self.image_to_annotations[img_name] = []
            self.image_to_annotations[img_name].append(ann)

        # Get unique image names
        self.image_names = list(self.image_to_annotations.keys())

        # Create category to index mapping
        self.categories = sorted(set(ann['category'] for ann in self.annotations))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

        # Store metadata
        self.metadata = {
            'weather': sorted(set(ann['weather'] for ann in self.annotations)),
            'scene': sorted(set(ann['scene'] for ann in self.annotations)),
            'timeofday': sorted(set(ann['timeofday'] for ann in self.annotations))
        }

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = self.images_dir / f"{img_name}.jpg"

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Get annotations for this image
        annotations = self.image_to_annotations[img_name]

        # Prepare boxes and labels
        boxes = []
        labels = []
        metadata = {
            'weather': [],
            'scene': [],
            'timeofday': []
        }

        for ann in annotations:
            # Convert bounding box to [x1, y1, x2, y2] format
            box = [ann['x1'], ann['y1'], ann['x2'], ann['y2']]
            boxes.append(box)

            # Convert category to index
            label = self.category_to_idx[ann['category']]
            labels.append(label)

            # Store metadata
            metadata['weather'].append(ann['weather'])
            metadata['scene'].append(ann['scene'])
            metadata['timeofday'].append(ann['timeofday'])

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
            'metadata': metadata
        }

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def get_category_names(self):
        """Return the list of category names"""
        return self.categories

    def get_num_categories(self):
        """Return the number of unique categories"""
        return len(self.categories)

    def get_metadata(self):
        """Return the metadata information"""
        return self.metadata

    def get_environment_info(self):
        """Return information about different environments in the dataset"""
        return {
            'weather': len(self.metadata['weather']),
            'scene': len(self.metadata['scene']),
            'timeofday': len(self.metadata['timeofday'])
        }