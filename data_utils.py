import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.datasets import CIFAR10





def get_normalized_data_loaders(train_dataset, test_dataset, batch_size, validation_split=0.1):
    # Compute the mean and standard deviation of the dataset
    train_mean = np.mean(train_dataset.data.astype(float), axis=(0,1,2)) / 255.0
    train_std = np.std(train_dataset.data.astype(float), axis=(0,1,2)) / 255.0
    print(f"Train mean: {train_mean}, Train std: {train_std}")
    
    # Normalize the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])
    
    # Apply the transform to the datasets
    train_dataset.transform = transform
    test_dataset.transform = transform
    
    # Split into train and validation sets
    val_size = int(len(train_dataset) * validation_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



def get_data_loaders(train_dataset, test_dataset, batch_size, validation_split=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Apply the transform to the datasets
    train_dataset.transform = transform
    test_dataset.transform = transform
    
    # Split into train and validation sets
    val_size = int(len(train_dataset) * validation_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader





# def get_cifar10_augmented_loaders(batch_size, validation_split=0.1):
#     DATA_DIR = '/datasets/cv_datasets/data'
#     ds_train = CIFAR10(root=DATA_DIR, download=True, train=True, transform=None)
#     ds_test = CIFAR10(root=DATA_DIR, download=True, train=False, transform=None)
    
    
#     transform_augmented = transforms.Compose([
#             transforms.RandomResizedCrop(32, scale=(0.7, 1.0)),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomApply(
#                 [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)],
#                 p=0.2
#             ),
#             transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784]),
#             transforms.ToTensor()
#         ])

#     transform_basic = transforms.Compose([
#         transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784]),
#         transforms.ToTensor()
#     ])

#     full_train = CIFAR10(root=DATA_DIR, train=True, download=True, transform=None)
#     test_set = CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_basic)

#     val_size = int(len(full_train) * validation_split)
#     train_size = len(full_train) - val_size
#     train_indices, val_indices = random_split(range(len(full_train)), [train_size, val_size])

#     train_set = Subset(CIFAR10(root=DATA_DIR, train=True, download=False, transform=transform_augmented), train_indices)
#     val_set = Subset(CIFAR10(root=DATA_DIR, train=True, download=False, transform=transform_basic), val_indices)

#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

#     return train_loader, val_loader, test_loader