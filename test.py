import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from torch.nn import init
from torch.utils.tensorboard import SummaryWriter
import os
from utils import save_checkpoint, accuracy

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_cifar100_data_loaders(download, path, shuffle=False, batch_size=256, choice='stl10'):
    if choice=='imagenet' or choice=="init":
        data_transforms = transforms.Compose([transforms.Resize(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif choice=="stl10":
        data_transforms = transforms.Compose([transforms.Resize(96),
                                            transforms.ToTensor(),
                                            ])
    elif choice=="cifar10":
        data_transforms = transforms.Compose([transforms.Resize(224),
                                            transforms.ToTensor(),
                                            ])

    train_dataset = datasets.CIFAR100(root=path, train=True, download=download,
                                    transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=4, drop_last=False, shuffle=shuffle)
    
    test_dataset = datasets.CIFAR100(root=path, train=False, download=download,
                                    transform=data_transforms)


    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                                num_workers=4, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


def load_model(path=None, pretrain=False):
    # Create model
    model = torchvision.models.resnet18(pretrained=pretrain)
    
    # freeze all layers except the last
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    num_classes = 100
    model.fc = torch.nn.Linear(512,num_classes)
    init.kaiming_normal_(model.fc.weight.data)
    # model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    if path is not None:
        # loading the trained check point data
        checkpoint = torch.load(path, map_location=device)

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]
        log = model.load_state_dict(state_dict, strict=False)
        assert log.missing_keys == ['fc.weight', 'fc.bias']

    model = model.to(device)
    return model


def test(model, data_path, choice):
    
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # optimize only the linear classifier
    
    cifar100_train_loader, cifar100_test_loader = get_cifar100_data_loaders(path=data_path, download=False, choice=choice)

    # supervise learning on CIFAR100 dataset

    top1_train_accuracy_list = [0]
    top1_accuracy_list = [0]
    top5_accuracy_list = [0]
    epoch_list = [0]

    top1_accuracy = 0
    top5_accuracy = 0
    model.eval()
    with torch.no_grad():
        for counter, (x_batch, y_batch) in enumerate(cifar100_test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
        
            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]
    
    top1_accuracy /= (counter + 1)
    top5_accuracy /= (counter + 1)

    top1_accuracy_list.append(top1_accuracy.item())
    top5_accuracy_list.append(top5_accuracy.item())
    print(f"Test \tTop1 Train accuracy \tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")



data_path = "datasets"
model_path = "slt10\model.pth.tar"
choice = 'stl10'
model = load_model(path=model_path,pretrain=False)
test(model, data_path, choice)
