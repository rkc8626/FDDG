import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

import matplotlib.pyplot as plt
from torchvision.utils import save_image
import json
import pandas as pd


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    "Debug28",
    "Debug224",
    "CCMNIST1",
    "FairFace",
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    "WILDSCamelyon",
    "WILDSFMoW",
    "Dataset100k",
    "BDD100kPerson"
]

def get_dataset_class(dataset_name):
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 8001
    CHECKPOINT_FREQ = 100
    N_WORKERS = 8
    ENVIRONMENTS = None
    INPUT_SHAPE = None

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']

class NYPD_OneEnv(Dataset):
    def __init__(self, env):
        df = pd.read_csv("/home/YOUR_PATH/data/NYPD/" + str(env) + ".csv", encoding='latin-1', low_memory=False)

        self.x, self.y, self.z = self.df2tensor(df)

    def df2tensor(self, initial_data):
        y = initial_data['frisked'].values
        others = initial_data.drop('frisked', axis=1)

        z = others['race_B'].values
        x = others.drop('race_B', axis=1).values

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y)
        z = torch.tensor(z, dtype=torch.float32)
        return x, y, z

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx]

class SensitiveImageFolder(ImageFolder):
    def __init__(self,
        root,
        transform = None,
        ):
        super().__init__(root,
                         transform,
                         )
        path_list = root.split('/')
        path_list.pop()
        dict_path = "/".join(path_list)
        with open(dict_path + '/data.json') as f:
            self.dict = json.load(f)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        file_name = path.split('/')[-1]

        z = self.dict[file_name][2] #
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, z

    def __len__(self):
        return  len(self.samples)

class NYPD(MultipleDomainDataset):
    N_WORKERS = 4
    CHECKPOINT_FREQ = 500
    def __init__(self, root, test_envs, hparams):
        super().__init__()

        self.datasets = []
        for i in range(5):
            self.datasets.append(NYPD_OneEnv(i))
        self.input_shape = [51]
        self.num_classes = 2


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()])

        augment_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = SensitiveImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class CCMNIST1(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = [0, 1, 2]
    def __init__(self, root, test_envs,hparams):
        self.dir = os.path.join("/home/YOUR_PATH/data/CCMNIST1/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class BDD100k(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 500
    # N_WORKERS = 4

class BDD100kPersonEnv(Dataset):
    def __init__(self, root_dir, json_file, transform=None):
        self.root_dir = root_dir
        with open(json_file, 'r') as f:
            self.annotations = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_name = ann['image_name']
        img_path = os.path.join(self.root_dir, f"{img_name}.jpg")

        # Load image and crop the person region
        image = Image.open(img_path).convert('RGB')
        # Crop the person region using bounding box
        x1, y1, x2, y2 = ann['x1'], ann['y1'], ann['x2'], ann['y2']
        image = image.crop((x1, y1, x2, y2))

        if self.transform:
            image = self.transform(image)

        # Get demographic attributes
        age = ann['age']      # age attribute (0/1)
        gender = ann['gender']  # gender attribute (0/1)
        skin = ann['skin']    # skin attribute (0/1)

        # You can modify which attribute to use as main task (y) and sensitive attribute (z)
        y = gender  # Using gender as the main task
        z = skin    # Using skin color as the sensitive attribute

        return image, y, z

class BDD100kPerson(MultipleDomainDataset):
    N_WORKERS = 4
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ['train', 'val', 'test']  # Different splits of the dataset
    INPUT_SHAPE = (3, 224, 224)

    def __init__(self, root, test_envs, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Base path for the BDD100k person dataset
        data_path = "/home/chenz1/toorange/Data/bdd100k_person"

        self.datasets = []

        # Create datasets for each environment using the same fair_labels.json
        # but different transforms for train/val/test
        for i, env in enumerate(self.ENVIRONMENTS):
            if hparams['data_augmentation'] and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = BDD100kPersonEnv(
                os.path.join(data_path, "images"),
                os.path.join(data_path, "fair_labels.json"),
                transform=env_transform
            )
            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224)
        self.num_classes = 2  # Binary classification (0/1)

class FairFace(MultipleEnvironmentImageFolder):
    N_WORKERS = 4
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ['0', '1', '2', '3', '4', '5', '6']
    def __init__(self, root, test_envs,hparams):
        self.dir = os.path.join("/home/YOUR_PATH/data/FairFace/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class YFCC(MultipleEnvironmentImageFolder):
    N_WORKERS = 4
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ['0', '1', '2']
    def __init__(self, root, test_envs,hparams):
        self.dir = os.path.join("/home/YOUR_PATH/data/YFCC/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 96, 96)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.RandomResizedCrop(96, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)

# class Dataset100kEnv(Dataset):
#     def __init__(self, root_dir, json_file, transform=None):
#         self.root_dir = root_dir
#         with open(json_file, 'r') as f:
#             self.annotations = json.load(f)
#         self.transform = transform

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         ann = self.annotations[idx]
#         img_name = ann['new_image_name']
#         img_path = os.path.join(self.root_dir, f"{img_name}.jpg")

#         # Load image
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)

#         # Get labels
#         category = ann['category']
#         weather = ann['weather']
#         scene = ann['scene']
#         timeofday = ann['timeofday']

#         # You can modify these mappings based on your needs
#         category_map = {'traffic sign': 0, 'traffic light': 1, 'car': 2}
#         weather_map = {'clear': 0, 'snowy': 1, 'rainy': 2, 'overcast': 3}
#         timeofday_map = {'daytime': 0, 'night': 1, 'dawn/dusk': 2}

# # y represents what object is in the image (traffic sign, traffic light, or car
#         y = category_map[category]  # Main task label
# # z represents the domain/environment condition (weather condition) under which the image was taken
#         z = weather_map[weather]    # Domain label (you can change this to scene or timeofday if needed)

#         return image, y, z

# class Dataset100k(MultipleDomainDataset):
#     N_WORKERS = 4
#     CHECKPOINT_FREQ = 300
#     ENVIRONMENTS = ['clear', 'snowy', 'rainy', 'overcast']  # Based on weather conditions
#     INPUT_SHAPE = (3, 224, 224)  # We'll resize all images to 224x224

#     def __init__(self, root, test_envs, hparams):
#         super().__init__()

#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])

#         # Paths for your dataset
#         data_path = "/home/chenz1/toorange/Data/100k"

#         # Create datasets for each environment (train, val, test)
#         self.datasets = []

#         # Training set
#         train_data = Dataset100kEnv(
#             os.path.join(data_path, "train_cropped"),
#             os.path.join(data_path, "train_labels.json"),
#             transform=transform
#         )
#         self.datasets.append(train_data)

#         # Validation set
#         val_data = Dataset100kEnv(
#             os.path.join(data_path, "val_cropped"),
#             os.path.join(data_path, "val_labels.json"),
#             transform=transform
#         )
#         self.datasets.append(val_data)

#         # Test set
#         test_data = Dataset100kEnv(
#             os.path.join(data_path, "test_cropped"),
#             os.path.join(data_path, "test_labels.json"),
#             transform=transform
#         )
#         self.datasets.append(test_data)

#         self.input_shape = (3, 224, 224)
#         self.num_classes = 3  # traffic sign, traffic light, car
