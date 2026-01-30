import torch, os, random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from collections import defaultdict
import numpy as np

def get_loader(dset, data_path, batch_size=128, num_workers=4, num_val=500, raw=False, target_sparse=False, u=0, noise=None, std=None, seed=11):
    os.makedirs(data_path, exist_ok=True)

    if dset == 's2m':
        svhn_tr = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        s_train = datasets.SVHN(os.path.join(data_path, 'svhn'), split='train', download=True, transform=svhn_tr)
        s_test = datasets.SVHN(os.path.join(data_path, 'svhn'), split='test', download=True, transform=svhn_tr)

        mnist_tr = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        t_train = datasets.MNIST(os.path.join(data_path, 'mnist'), train=True, download=True, transform=mnist_tr)
        t_test = datasets.MNIST(os.path.join(data_path, 'mnist'), train=False, download=True, transform=mnist_tr)

    elif dset == 'u2m':
        tr = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        s_train = datasets.USPS(os.path.join(data_path, 'usps'), train=True, download=True, transform=tr)
        s_test = datasets.USPS(os.path.join(data_path, 'usps'), train=False, download=True, transform=tr)

        t_train = datasets.MNIST(os.path.join(data_path, 'mnist'), train=True, download=True, transform=tr)
        t_test = datasets.MNIST(os.path.join(data_path, 'mnist'), train=False, download=True, transform=tr)

    elif dset == 'm2u':
        tr = transforms.Compose([
            transforms.Resize([32, 32]),
            #transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        s_train = datasets.MNIST(os.path.join(data_path, 'mnist'), train=True, download=True, transform=tr)
        s_test = datasets.MNIST(os.path.join(data_path, 'mnist'), train=False, download=True, transform=tr)

        t_train = datasets.USPS(os.path.join(data_path, 'usps'), train=True, download=True, transform=tr)
        t_test = datasets.USPS(os.path.join(data_path, 'usps'), train=False, download=True, transform=tr)

    elif dset == 'm2mm':
        mnist_tr = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        s_train = datasets.MNIST(os.path.join(data_path, 'mnist'), train=True, download=True, transform=mnist_tr)
        s_test = datasets.MNIST(os.path.join(data_path, 'mnist'), train=False, download=True, transform=mnist_tr)

        mnistm_tr = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        t_train = datasets.ImageFolder(root=os.path.join(data_path, 'mnistm', 'trainset'), transform=mnistm_tr)
        t_test = datasets.ImageFolder(root=os.path.join(data_path, 'mnistm', 'testset'), transform=mnistm_tr)

    else:
        raise ValueError(f"Unsupported dataset: {dset}")

    
    t_train_size = len(t_train) - num_val
    t_train_data, t_val_data = random_split(t_train, [t_train_size, num_val], generator=torch.Generator().manual_seed(42))

    if target_sparse:
        random.seed(seed) 
        label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(t_train_data):
            label_to_indices[label].append(idx)

        selected_indices = []
        for label, indices in label_to_indices.items():
            if len(indices) < u:
                raise ValueError(f"Not enough samples for label {label} to extract {u} samples.")
            selected_indices.extend(random.choices(indices, k=u))

        t_train_data = Subset(t_train_data, selected_indices)

    if raw:
        
        return s_train, s_test, t_train_data, t_val_data, t_test
    else:
        
        s_train_loader = DataLoader(s_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        s_test_loader = DataLoader(s_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

        t_train_loader = DataLoader(t_train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True) #check
        t_val_loader = DataLoader(t_val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        target_loader = DataLoader(t_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
        
        if noise is None:
            t_test_loader = DataLoader(t_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        else:
            t_test_loader = DataLoader(t_test_noise, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
            
        return s_train_loader, s_test_loader, t_train_loader, t_val_loader, t_test_loader, target_loader
