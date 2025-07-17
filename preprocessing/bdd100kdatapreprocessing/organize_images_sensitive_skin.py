# organize_images_sensitive_gender.py
# Classify: isPerson
# Sensitive: dark, light, unlabelled
# Domain: Daytime, Darktime

import json
import os
import shutil
from tqdm import tqdm

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, '/home/chenz1/toorange/Data/new_8k_520/img_mixed')
TARGET_DIR = os.path.join(BASE_DIR, '/home/chenz1/toorange/Data/new_8k_520/processed_8k_skin')
LABELS_FILE = os.path.join(BASE_DIR, '/home/chenz1/toorange/Data/new_8k_520/labels_mixed.json')

print(f"Base directory: {BASE_DIR}")
print(f"Source directory: {SOURCE_DIR}")
print(f"Target directory: {TARGET_DIR}")
print(f"Labels file: {LABELS_FILE}")

def create_directory_structure():
    """Create the hierarchical directory structure."""
    timeofday_categories = ['daytime', 'darktime']
    person_categories = ['person', 'non_person']

    print("Creating directory structure...")
    for timeofday in tqdm(timeofday_categories, desc="Creating directories"):
        for person in person_categories:
            path = os.path.join(TARGET_DIR, timeofday, person)
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

def create_annotation_file(image_count, mapping_dict, skipped_count):
    """Create an annotation file explaining the dataset structure."""
    print("Creating annotation file...")
    with tqdm(total=1, desc="Writing annotation file") as pbar:
        annotation = {
            "dataset_info": {
                "name": "BDD100K Person Dataset",
                "description": "A processed subset of BDD100K dataset focusing on person detection and attributes",
                "total_images": image_count,
                "skipped_images": skipped_count,
                "skipped_reason": "Images not found in source directory were skipped",
                "structure": {
                    "directory_structure": {
                        "processed_new/": {
                            "daytime/": {
                                "person/": "Contains images with persons in daytime",
                                "non_person/": "Contains images without persons in daytime",
                            },
                            "darktime/": {
                                "person/": "Contains images with persons in darktime/night/dawn/dusk",
                                "non_person/": "Contains images without persons in darktime/night/dawn/dusk",
                            }
                        }
                    },
                    "file_structure": {
                        "image_folders": "Each subfolder contains numerically named images (0.jpg, 1.jpg, etc.)",
                        "data.json": "Contains attributes for all images in format {image_id: [isperson, skin, timeofday]}",
                        "annotation.json": "This file - contains dataset information and image mappings"
                    }
                },
                "attributes": {
                    "isperson": {
                        "1": "Person present",
                        "0": "No person present"
                    },
                    "skin": {
                        "0": "Light",
                        "1": "Dark",
                        "": "Not labeled"
                        # -1 is unknown for skin, but not used in this dataset
                    },
                    "timeofday": {
                        "daytime": "Daytime images",
                        "darktime": "Night/dark time/dawn/dusk images",
                        "unknown": "Unknown time of day"
                    }
                }
            },
            "image_mapping": mapping_dict
        }

        annotation_path = os.path.join(TARGET_DIR, 'annotation.json')
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=4)
        pbar.update(1)
    print(f"Created annotation file with dataset information. Skipped {skipped_count} images not found in source directory.")

def organize_images():
    """Organize images based on timeofday and person presence."""
    # Read labels file
    print("Reading labels file...")
    with tqdm(total=1, desc="Loading labels") as pbar:
        with open(LABELS_FILE, 'r') as f:
            labels = json.load(f)
        pbar.update(1)

    # Create a mapping for quick lookup
    print("Processing labels...")
    image_mapping = {}
    data_json = {}
    image_counter = 0
    skipped_counter = 0
    original_to_new_mapping = {}
    category_counter = {}
    non_person_label_names = []

    for item in tqdm(labels, desc="Creating image mapping"):
        category = item['category'] if 'category' in item else 'unknown'
        # Robustly handle skin: treat None, 'null', or missing as null, -1 as light, 1 as dark
        skin = item.get('skin', None)
        if skin is None or skin == 'null':
            skin = -1
        else:
            try:
                skin = int(skin)
            except (ValueError, TypeError):
                skin = None
            else:
                if skin == 0:
                    skin = 0  # Map 0 to -1 for light
                elif skin == 1:
                    skin = 1   # 1 stays as dark
                else:
                    skin = -1  # Any other value is treated as null
        # Construct image_name according to category
        if category == 'person':
            image_name = f"{item['image_name']}_{item['object_id']}.jpg"
        else:
            image_name = f"{item['image_name']}_{category}_{item['object_id']}.jpg"
        image_mapping[image_name] = {
            'timeofday': item['timeofday'] if item['timeofday'] is not None else 'unknown',
            'category': category,
            'skin': skin
        }
        category_counter[category] = category_counter.get(category, 0) + 1
        if category != 'person':
            non_person_label_names.append(image_name)

    print("Category distribution in labels:", category_counter)
    print(f"Sample non-person image_names from labels: {non_person_label_names[:10]}")

    # Get list of all images
    image_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.jpg')]
    print(f"Total images in source: {len(image_files)}")
    print(f"Sample image files: {image_files[:5]}")
    # Check which non-person label image_names exist in source
    non_person_in_source = [name for name in non_person_label_names if name in image_files]
    print(f"Non-person image_names in labels: {len(non_person_label_names)}")
    print(f"Non-person image_names that exist in source: {len(non_person_in_source)}")
    print(f"Sample non-person image_names that exist in source: {non_person_in_source[:10]}")

    # Process each image with progress bar
    print("\nProcessing and copying images...")
    non_person_count = 0
    for image_file in tqdm(image_files, desc="Processing images"):
        if image_file in image_mapping:
            attrs = image_mapping[image_file]
            timeofday = attrs['timeofday']
            is_person = 1 if attrs['category'] == 'person' else 0
            skin = attrs['skin']

            if is_person == 0:
                non_person_count += 1
                # print(f"Non-person image: {image_file}, category: {attrs['category']}")

            # Handle timeofday categorization
            if timeofday == 'darktime':
                timeofday = 'darktime'
            elif timeofday == 'daytime':
                timeofday = 'daytime'

            timeofday_binary = 1 if timeofday == 'daytime' else 0  # Simplified since we only have daytime/darktime

            # Create new numerical filename
            new_filename = f"{image_counter}.jpg"
            original_to_new_mapping[new_filename] = {
                "original_name": image_file,
                "timeofday": timeofday,
                "isperson": is_person,
                "category": attrs['category'],  # Store the original category for reference
                "attributes": {
                    "skin": skin
                }
            }

            # Create directories if they don't exist
            person_category = "person" if is_person == 1 else "non_person"
            target_dir = os.path.join(TARGET_DIR, timeofday, person_category)
            os.makedirs(target_dir, exist_ok=True)

            source_path = os.path.join(SOURCE_DIR, image_file)
            target_path = os.path.join(target_dir, new_filename)

            # Copy the image to appropriate directory
            shutil.copy2(source_path, target_path)

            # Add to data.json
            data_json[new_filename] = [
                timeofday_binary,
                is_person,
                skin
            ]

            image_counter += 1
        else:
            skipped_counter += 1

    # Create main data.json in the processed directory
    print("\nCreating data.json...")
    with tqdm(total=1, desc="Writing data.json") as pbar:
        data_json_path = os.path.join(TARGET_DIR, 'data.json')
        with open(data_json_path, 'w') as f:
            json.dump(data_json, f, indent=4)
        pbar.update(1)

    # Create annotation file
    create_annotation_file(image_counter, original_to_new_mapping, skipped_counter)

    print(f"Total non-person images processed: {non_person_count}")

def main():
    print("Starting image organization process...")
    create_directory_structure()
    organize_images()
    print("\nImage organization completed successfully!")

if __name__ == "__main__":
    main()