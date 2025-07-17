# organize_images_sensitive_gender.py
# Classify: isPerson
# Sensitive: weather
#    3091 clear (good weather = 0)
#    1808 undefined (skip)
#    1191 overcast (bad weather = 1)
#     664 snowy (bad weather = 1)
#     511 rainy (bad weather = 1)
#     464 partly cloudy (bad weather = 1)
# Domain: Daytime, Darktime

import json
import os
import shutil
from tqdm import tqdm

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, '/home/chenz1/toorange/Data/new_8k_520/img_mixed')
TARGET_DIR = os.path.join(BASE_DIR, '/home/chenz1/toorange/Data/new_8k_520/processed_8k_weather')
LABELS_FILE = os.path.join(BASE_DIR, '/home/chenz1/toorange/Data/new_8k_520/labels_mixed_weather_binary.json')

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
    non_person_count = 0

    for item in tqdm(labels, desc="Creating image mapping"):
        category = item['category'] if 'category' in item else 'unknown'
        # Get weather and convert to binary classification
        weather = item.get('weather', 'undefined')
        if weather is None or weather == 'null':
            continue
        weather_str = str(weather).strip().lower()

        # Direct binary weather classification
        if weather_str == 'goodweather':
            weather_code = 0  # good weather
        elif weather_str in ['badweather']:
            weather_code = 1  # bad weather
        else:
            continue  # Skip any other weather conditions

        # Construct image_name according to category
        if category == 'person':
            image_name = f"{item['image_name']}_{item['object_id']}.jpg"
        else:
            image_name = f"{item['image_name']}_{category}_{item['object_id']}.jpg"
        image_mapping[image_name] = {
            'timeofday': item['timeofday'] if item['timeofday'] is not None else 'unknown',
            'category': category,
            'weather': weather_code  # Binary: 0 for good weather, 1 for bad weather
        }
        category_counter[category] = category_counter.get(category, 0) + 1
        if category != 'person':
            non_person_label_names.append(image_name)

    print("Category distribution in labels:", category_counter)
    # Get list of all images
    image_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.jpg')]

    # Process each image with progress bar
    print("\nProcessing and copying images...")
    undefined_weather_count = 0  # Counter for images with weather 'undefined'
    for image_file in tqdm(image_files, desc="Processing images"):
        if image_file in image_mapping:
            attrs = image_mapping[image_file]
            timeofday = attrs['timeofday']
            is_person = 1 if attrs['category'] == 'person' else 0
            weather = attrs['weather']

            if is_person == 0:
                non_person_count += 1

            # Handle timeofday categorization
            if timeofday == 'darktime':
                timeofday = 'darktime'
            elif timeofday == 'daytime':
                timeofday = 'daytime'

            timeofday_binary = 1 if timeofday == 'daytime' else 0

            # Create new numerical filename
            new_filename = f"{image_counter}.jpg"
            original_to_new_mapping[new_filename] = {
                "original_name": image_file,
                "timeofday": timeofday,
                "isperson": is_person,
                "category": attrs['category'],
                "attributes": {
                    "weather": weather  # Binary: 0 for good weather, 1 for bad weather
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
                weather  # Binary: 0 for good weather, 1 for bad weather
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
    print(f"Total images skipped due to undefined weather: {undefined_weather_count}")

def create_annotation_file(image_counter, original_to_new_mapping, skipped_counter):
    """Create annotation file with mapping information."""
    annotation_path = os.path.join(TARGET_DIR, 'annotation.json')
    with open(annotation_path, 'w') as f:
        json.dump({
            'total_images': image_counter,
            'skipped_images': skipped_counter,
            'mapping': original_to_new_mapping
        }, f, indent=4)

def main():
    print("Starting image organization process...")
    create_directory_structure()
    organize_images()
    print("\nImage organization completed successfully!")

if __name__ == "__main__":
    main()