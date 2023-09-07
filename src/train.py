import os
import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

from utils.confusion_matrix import ConfusionMatrix

def set_model(num_classes):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(train_loader, test_loader, model_path, weight_path, image_size, device):
    classes = train_loader.dataset.classes
    num_classes = len(classes)
    print(num_classes)

    cm = ConfusionMatrix(classes=classes)
    model = set_model(num_classes=num_classes)
    summary(model, (3, *image_size))

    
    # model_file_name = "./model.pt"
    if model_path is not None:
        assert not os.path.isfile(model_path), "model.pt file already exists."
        torch.save(model, model_path)
        print("[INFO] Save model file to ", os.path.abspath(model_path))

    model = model.to(device=device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    train_iter = iter(train_loader)
    num_batchs = 0
    best_acc = 0
    while(True):
        try:
            batch = next(train_iter)
        except:
            train_iter = iter(train_loader)
        num_batchs += 1

        image, target = batch
        image = image.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        
        if num_batchs % 100 == 0:
            print(f"[INFO] Number of batchs: {num_batchs}, Training loss: {loss.item():.3f}")

            with torch.no_grad():
                model.eval()
                eval_loss = 0
                for  batch in test_loader:
                    image, target = batch
                    image = image.to(device)
                    target = target.to(device)
                    pred = model(image)
                    loss = criterion(pred, target)

                    _, act = torch.max(pred, 1)
                    cm.update(act.cpu().numpy(), target.cpu().numpy())

                acc = cm.get_accuracy()
                if acc > best_acc and weight_path is not None:
                    torch.save({
                        'num_batchs': num_batchs,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'acc':cm.get_accuracy(),
                        'acc_per_cls': {**cm.get_accuracy_per_cls()}}, 
                        weight_path)
                    print(f"[INFO] Save weight file to {os.path.abspath(weight_path)}")
                    best_acc = acc
                cm.print_calc()
                cm.clear()

if __name__ == "__main__":
    train()