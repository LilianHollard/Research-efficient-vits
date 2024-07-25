import torch

import torchvision
import torchvision.transforms as transforms


from torch.utils.data.distributed import DistributedSampler

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


IMGNET_TRAIN_MEAN = (0.485, 0.456, 0.406)
IMGNET_TRAIN_STD = (0.229, 0.224, 0.225)


transform = transforms.Compose([
    transforms.Resize((224,224)),
    #transforms.RandomCrop(224), #weird results with random crop on validation (which I guess is due to the randomness of the cropping within the 320 by 320px of ImageNette image size)
    transforms.ToTensor(),
    transforms.Normalize(IMGNET_TRAIN_MEAN, IMGNET_TRAIN_STD)
])

train_aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    #transforms.Resize((224,224)),
    transforms.RandomCrop(224, padding=4),
    # transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(IMGNET_TRAIN_MEAN, IMGNET_TRAIN_STD),
    # transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
])


transform_high = transforms.Compose([
    transforms.ColorJitter(0.4), #from LeVIT data color giter transform
    transforms.AutoAugment(), #default is ImageNet auto augment policy (but I don't know if it really working or not)
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(IMGNET_TRAIN_MEAN, IMGNET_TRAIN_STD)
])


#EfficientViT paper
#Mixup
#Auto-augmentation
#random erasing

#note : v2 is required for MixUp augmentation
effvit_transforms = transforms.Compose([
    transforms.AutoAugment(),
#    transforms.RandomErasing(), doesnt work ?
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(IMGNET_TRAIN_MEAN, IMGNET_TRAIN_STD),
])

def get_data(path, batch_size=32, ddp=False, num_replicas=None, rank=None):
    trainset = torchvision.datasets.ImageFolder(path+"/train/",
                                                transform=effvit_transforms)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size, shuffle=True, num_workers=8, pin_memory=True)

    if ddp:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=False, num_workers=8, pin_memory=True,
                                                  sampler=DistributedSampler(trainset, num_replicas=num_replicas,
                                                                             rank=rank))

    testset = torchvision.datasets.ImageFolder(path+"/val/",
                                               transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size * 2, pin_memory=True, num_workers=4)

    return trainloader, testloader