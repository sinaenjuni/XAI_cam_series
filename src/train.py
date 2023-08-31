import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision import models
from torchsummary import summary

from utils.confusion_matrix import ConfusionMatrix
from datasets.cifar10 import CIFAR10


def set_data(data_path, batch_size, image_size):
    transforms = Compose([ToTensor(), 
                          Resize(size=image_size, antialias=False), 
                          Normalize(mean=(.5,.5,.5), std=(.5,.5,.5))])
    
    trainset = CIFAR10(root=data_path, train=True, download=True, transform=transforms, num_batchs=batch_size*2000)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    testset = CIFAR10(root=data_path, train=False, download=True, transform=transforms)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    return train_loader, test_loader

def set_model(num_classes):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train():
    if torch.has_mps:
        device = torch.device("mps")

    data_path = "data/"
    batch_size = 32
    image_size = (256, 256)

    train_loader, test_loader = set_data(data_path=data_path, batch_size=batch_size, image_size=image_size)
    classes = train_loader.dataset.classes
    num_classes = len(classes)
    print(num_classes)

    cm = ConfusionMatrix(classes=classes)
    model = set_model(num_classes=num_classes)
    summary(model, (3, *image_size))

    torch.save(model, "./model.pth")
    model = model.to(device=device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    train_iter = iter(train_loader)
    
    num_epochs = 0
    num_batchs = 0
    best_loss = float("inf")
    while(True):
        try:
            batch = next(train_iter)
        except:
            train_iter = iter(train_loader)
            num_epochs += 1    
        num_batchs += 1
        print(num_epochs, ", ", num_batchs, sep="")

        image, target = batch
        image = image.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, act = torch.max(pred, 1)
            cm.update(act.cpu().numpy(), target.cpu().numpy())

        if num_batchs % 100 == 0 and best_loss > loss.item():
            cm.print_calc()
            torch.save({
                        'num_batchs': num_batchs,
                        'num_epochs': num_epochs,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'acc':cm.get_accuracy(),
                        'acc_per_cls': {**cm.get_accuracy_per_cls()}}, 
                        f"./weights_{num_batchs}.pt")
            best_loss = loss.item()
            cm.clear()
            print("Save succsess!!")

if __name__ == "__main__":
    train()