import json
import os
import shutil
from tqdm import tqdm

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, '/home/chenz1/toorange/Data/new_8k_520/img_mixed')
TARGET_DIR = os.path.join(BASE_DIR, '/home/chenz1/toorange/Data/new_8k_520/processed_8k_520')
LABELS_FILE = os.path.join(BASE_DIR, '/home/chenz1/toorange/Data/new_8k_520/labels_mixed.json')

print(f"Base directory: {BASE_DIR}")
print(f"Source directory: {SOURCE_DIR}")
print(f"Target directory: {TARGET_DIR}")
print(f"Labels file: {LABELS_FILE}")

def create_directory_structure():
    """Create the hierarchical directory structure."""
    timeofday_categories = ['daytime', 'darktime']
    age_categories = ['age0', 'age1']
     
    print("Creating directory structure...")
    for timeofday in tqdm(timeofday_categories, desc="Creating directories"):
        for age in age_categories:
            path = os.path.join(TARGET_DIR, timeofday, age)
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
                                "age0/": "Contains images of age group 0 in daytime",
                                "age1/": "Contains images of age group 1 in daytime",
                            },
                            "darktime/": {
                                "age0/": "Contains images of age group 0 in darktime/night/dawn/dusk",
                                "age1/": "Contains images of age group 1 in darktime/night/dawn/dusk",
                            }
                        }
                    },
                    "file_structure": {
                        "image_folders": "Each subfolder contains numerically named images (0.jpg, 1.jpg, etc.)",
                        "data.json": "Contains attributes for all images in format {image_id: [gender, age, skin]}",
                        "annotation.json": "This file - contains dataset information and image mappings"
                    }
                },
                "attributes": {
                    "gender": {
                        "0": "Male",
                        "1": "Female",
                        "-1": "Unknown"
                        # -1 is unknown for gender, but not used in this dataset
                    },
                    "age": {
                        "0": "Adult",
                        "1": "Child",
                        "-1": "Unknown"
                        # -1 is unknown for age, but not used in this dataset
                    },
                    "skin": {
                        "0": "Light",
                        "1": "Dark",
                        "null": "Not labeled"
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
    """Organize images based on timeofday and age attributes."""
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

    for item in tqdm(labels, desc="Creating image mapping"):
        image_name = f"{item['image_name']}_{item['object_id']}.jpg"
        image_mapping[image_name] = {
            'timeofday': item['timeofday'] if item['timeofday'] is not None else 'unknown',
            'age': item['age'] if item['age'] is not None else 'unknown',
            'gender': item['gender'] if item['gender'] is not None else 'unknown',
            'skin': item['skin']
        }

    # Get list of all images
    image_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.jpg')]

    # Process each image with progress bar
    print("\nProcessing and copying images...")
    for image_file in tqdm(image_files, desc="Processing images"):
        if image_file in image_mapping:
            attrs = image_mapping[image_file]
            timeofday = attrs['timeofday']
            age = attrs['age']

            # Handle timeofday categorization
            if timeofday == 'darktime':
                timeofday = 'darktime'
            elif timeofday == 'daytime':
                timeofday = 'daytime'

            # Handle age categorization
            if age == 'unknown':
                age = 'unknown'

            timeofday_binary = 1 if timeofday == 'daytime' else 0  # Simplified since we only have daytime/darktime
            age_binary = int(age) if age != 'unknown' else -1

            # Create new numerical filename
            new_filename = f"{image_counter}.jpg"
            original_to_new_mapping[new_filename] = {
                "original_name": image_file,
                "timeofday": timeofday,
                "age": age,
                "attributes": {
                    "gender": attrs['gender'],
                    "age": age_binary,
                    "skin": attrs['skin']
                }
            }

            # Create directories if they don't exist
            target_dir = os.path.join(TARGET_DIR, timeofday, f"age{age}")
            os.makedirs(target_dir, exist_ok=True)

            source_path = os.path.join(SOURCE_DIR, image_file)
            target_path = os.path.join(target_dir, new_filename)

            # Copy the image to appropriate directory
            shutil.copy2(source_path, target_path)

            # Add to data.json
            data_json[new_filename] = [
            timeofday_binary,
            age_binary,
            -1 if attrs['gender'] == 'unknown' else attrs['gender']
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

def main():
    print("Starting image organization process...")
    create_directory_structure()
    organize_images()
    print("\nImage organization completed successfully!")

if __name__ == "__main__":
    main()