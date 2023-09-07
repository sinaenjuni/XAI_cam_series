import argparse
import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.utils.data import DataLoader
from datasets.cifar10 import CIFAR10

from train import train
from cam import cam
from grad_cam import grad_cam
from grad_campp import grad_campp

def parse_args():
    parser = argparse.ArgumentParser(
            description='Simple model training funtion for Clasee Activation Map(CAM) serise')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['train', 'cam', 'grad-cam', 'grad-cam++'], 
                        help='Mode select')
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='Save path for model.pt file')
    parser.add_argument('--weight_path', type=str, required=False, default=None,
                        help='Save path for weight.pt file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='data path for model training or \
                            Class Activation Map function')
    parser.add_argument('--batch_size', type=int, required=True, default=32,
                    help='data path for model training or \
                        Class Activation Map function')
    parser.add_argument('--image_size', type=int, required=True, default=256,
                    help='data path for model training or \
                        Class Activation Map function')
    parser.add_argument('--num_batchs', type=int, required=False, default=10000,
                    help='Number of pseudo batch size for training iteration')
    parser.add_argument('--device', type=str, required=True,
                        choices=['cpu', 'cuda', 'mps'], 
                        help='device select')

    args = parser.parse_args()
    return args


class SetDataset:
    def __init__(self, data_path, image_size, batch_size, num_batchs=20000) -> None:
        self.mean=(.5,.5,.5)
        self.std=(.5,.5,.5)
        self.image_size=image_size
        self.data_path=data_path
        self.batch_size=batch_size

        transforms = Compose([ToTensor(), 
                            Resize(size=image_size, antialias=False), 
                            Normalize(mean=self.mean, std=self.std)])
        
        trainset = CIFAR10(root=self.data_path, train=True, download=True, transform=transforms, num_batchs=self.batch_size*num_batchs)
        self.train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
        testset = CIFAR10(root=self.data_path, train=False, download=True, transform=transforms)
        self.test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    def get_train_loader(self):
        return self.train_loader
    def get_test_loader(self):
        return self.test_loader

if __name__ == "__main__":
    args = parse_args()
    print("mode:", args)
    device = torch.device(args.device)
    args.image_size = (args.image_size, args.image_size)
    set_dataset = SetDataset(data_path=args.data_path, 
                            batch_size=args.batch_size, 
                            image_size=args.image_size)
    train_loader = set_dataset.get_train_loader()
    test_loader = set_dataset.get_test_loader()

    if args.mode == 'train':
        train(train_loader=train_loader,
              test_loader=test_loader,
              model_path=args.model_path,
              image_size=args.image_size,
              weight_path=args.weight_path,
              device=device)
        
    elif args.mode == 'cam':
        cam(model_path=args.model_path, 
            weight_path=args.weight_path, 
            image_size=args.image_size, 
            train_loader=train_loader, 
            mean=set_dataset.mean,
            std=set_dataset.std,
            device=device)
        
    elif args.mode == 'grad-cam':
        grad_cam(model_path=args.model_path, 
            weight_path=args.weight_path, 
            image_size=args.image_size, 
            train_loader=train_loader, 
            mean=set_dataset.mean,
            std=set_dataset.std,
            device=device)
    elif args.mode == 'grad-cam++':
        grad_campp(model_path=args.model_path, 
            weight_path=args.weight_path, 
            image_size=args.image_size, 
            train_loader=train_loader, 
            mean=set_dataset.mean,
            std=set_dataset.std,
            device=device)
        
