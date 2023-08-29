import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
from PIL import Image 
from torchsummary import summary
from datasets.cifar10 import CIFAR10
from utils.confision_matrix import ConfusionMatrix


# mean = torch.as_tensor(data=(.5, .5 ,.5), dtype=torch.float32).view(-1, 1, 1)
# std = torch.as_tensor(data=(.5, .5 ,.5), dtype=torch.float32).view(-1, 1, 1)

class Np2Pt:
    def __init__(self) -> None:
        pass
    def __call__(self, np_tensor):
        return torch.from_numpy(np_tensor)
class MyDataset(Dataset):
    @staticmethod
    def get_transforms():
        return Compose([ToTensor(), # dvide 255., permute (C, H, W), and convert dtype to torch.float32
                        Resize(size=(256,256), antialias=False),
                        # Normalize((.5,.5,.5), (.5,.5,.5))
                        ])
    @staticmethod
    def load_data():
        # data = np.load("/Users/shinhyeonjun/code/class_activation_map/data/Imagenet64_val_npz/val_data.npz")
        data = np.load("/Users/shinhyeonjun/code/class_activation_map/data/Imagenet64_train_part1_npz/train_data_batch_1.npz")
        images = data['data'].reshape(-1, 3, 64, 64)
        images = images.transpose(0, 2, 3, 1)
        targets = data['labels']
        return images, targets
    def __init__(self):
        super(MyDataset).__init__()
        self.images, self.targets = self.load_data()
        self.transforms = self.get_transforms()

    def __getitem__(self, index):
        img = self.images[index]
        target = self.targets[index]
        img = self.transforms(img)
        return img, target
    def __len__(self):
        return len(self.images)

import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

if __name__ == "__main__":
    if torch.has_mps:
        device = torch.device("mps")

    data_path = "/Users/shinhyeonjun/code/class_activation_map/data/"
    batch_size = 32
    image_size = (256, 256)
    transforms = Compose([ToTensor(), 
                          Resize(size=image_size, antialias=False), 
                          Normalize(mean=(.5,.5,.5), std=(.5,.5,.5))])

    trainset = CIFAR10(root=data_path, train=True, download=True, transform=transforms, num_batchs=batch_size*2000)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    testset = CIFAR10(root=data_path, train=False, download=True, transform=transforms)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    # train_loader, test_laoder = set_dataset()
    classes = trainset.classes
    num_classes = len(classes)
    cm = ConfusionMatrix(classes=classes)

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    summary(model, (3, *image_size))
    torch.save(model, "./model.pth")
    model = model.to(device=device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    

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

        
        # if loss.item() < 1.0:
            # print("stop")
        
    _, act = torch.max(pred, 1)
    cm = np.zeros(size=(num_classes, num_classes))

    for a, t in zip(act, target):
        cm[a][t] += 1

    [a ]

    # for epoch in range(10):
    #     train_loss = []
    #     train_pred = []
    #     train_target = []
    #     for batch_ind, batch in enumerate(train_loader):
    #         image, target = batch
    #         image = image.to(device)
    #         target = target.to(device)

    #         optimizer.zero_grad()
    #         pred = model(image)
    #         loss = criterion(pred, target)
    #         loss.backward()
            
    #         with torch.no_grad():
    #             _, pred = pred.max(1)
    #             train_loss += [loss.cpu()]
    #             train_pred += [pred.cpu()]
    #             train_target += [target.cpu()]
            
    #         # print(loss.item(), pred.eq(target).sum().item())


    #     print(epoch)
    #     print(torch.mean(torch.stack(train_loss)).item())
    #     print((torch.mean((torch.concat(train_pred) == torch.concat(train_target)).to(torch.float16)) * 100).item())

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # sample_image, sample_target = next(iter(train_loader))





    # dataset = MyDataset()
    # data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)
    # # list(data_loader)[0]
    
    # # timg, ttarget = next(iter(data_loader))
    # # timg, ttarget = timg.to(device), ttarget.to(device)

    # # model.to(device)

    # targets = []
    # preds = []

    # for i, (image, target) in enumerate(data_loader):
    #     with torch.no_grad():
    #         image = image.to(device)

    #         pred = model(image)
    #         pred = torch.nn.functional.softmax(pred, 1)
    #         pred = pred.argmax(1)

    #         targets += [target]
    #         preds += [pred.cpu()]
    #     print(preds)

            

        


# img, target = mydata[1]
# print(img.shape)

# data = np.load("../data/Imagenet64_val_npz/val_data.npz")
# print(images.shape)
# image = images[1].transpose(1,2,0)
# # print(image.shape)
# pilimage = Image.fromarray(img.permute(1,2,0).numpy()+0.5*255)
# pilimage.show()


# print(torch.has_mps)
# print(models.VGG16_Weights.IMAGENET1K_V1)


# print(model)

# print("hello") 