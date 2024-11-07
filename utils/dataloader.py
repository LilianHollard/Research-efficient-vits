import torch
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

import numpy as np


# Transformations and data loading
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

effvit_transforms = transforms.Compose([
    AutoAugment(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])



def add_noise_dataset(dataset, noise_ratio=0.1):
    num_samples = len(dataset)
    
    num_noisy = int(noise_ratio * num_samples)
    
    labels = np.array(dataset.targets)
    
    #select index to modify
    noisy_indices = np.random.choice(num_samples, num_noisy, replace=False)
    
    for idx in noisy_indices:
        current_label = labels[idx]
        #give random value between 0 and 9
        new_label = np.random.choice([i for i in range(10) if i != current_label])
        labels[idx] = new_label
    
    dataset.targets = labels.tolist()
    
        


def add_noise(labels, noise):
    size = labels.size() 
    num_random_elements = int(labels.numel() * noise)
    #generate random index
    indices = torch.randperm(labels.numel())[:num_random_elements]
    tensor_flat = labels.view(-1)

    # Remplacer les valeurs aux indices sélectionnés par des valeurs différentes
    for idx in indices:
        current_value = tensor_flat[idx]
        new_value = torch.randint(0, 9, (1,)).item()  # Générer une valeur différente
        # Boucler jusqu'à ce que la nouvelle valeur soit différente de l'actuelle
        while new_value == current_value:
            new_value = torch.randint(0, 9, (1,)).item()
        tensor_flat[idx] = new_value

    # Reshaper le tensor pour revenir à sa taille d'origine
    tensor = tensor_flat.view(size)
    return tensor

def get_data(path, batch_size=32, ddp=False, num_replicas=None, rank=None, noise=0.0):
    trainset = torchvision.datasets.ImageFolder(path+"/train/",
                                                transform=effvit_transforms)
    
    
    if noise > 0.0:
        add_noise_dataset(trainset, noise)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size, shuffle=True, num_workers=8, pin_memory=True)

    if ddp:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=False, num_workers=8, pin_memory=True,
                                                  sampler=DistributedSampler(trainset, num_replicas=num_replicas,
                                                                             rank=rank))

    testset = torchvision.datasets.ImageFolder(path+"/val/",
                                               transform=test_transforms)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size//2, pin_memory=True, num_workers=8)

    return trainloader, testloader


def get_cifar(path, batch_size=32, ddp=False, num_replicas=None, rank=None, noise=0.0):
    train_dataset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform_aug)
    test_dataset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
    
    if noise > 0.0:
        add_noise_dataset(train_dataset, noise)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=8,  pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size//2, shuffle=False, num_workers=8, pin_memory=True)
    
    if ddp:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True,
                                             sampler=DistributedSampler(train_dataset, num_replicas=num_replicas, rank=rank))
        
        
    
    
    return train_loader, test_loader
        
   
