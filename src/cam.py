import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from datasets.cifar10 import CIFAR10
from torch.utils.data import DataLoader
from torchsummary import summary
# from PIL import Image as pilimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

from torchvision.transforms.functional import resize, InterpolationMode

class Cam(torch.nn.Module):
    def __init__(self, model, image_size):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.weights = self.model.fc.weight # weigth shape is [classes, features]

        for module_name, module in self.model.named_modules():
            if isinstance(module, torch.nn.modules.pooling.AdaptiveAvgPool2d):
                module.register_forward_hook(self.hook)
            
    def hook(self, module, input, output):
        self.features = input[0]

    def forward(self, images):
        n, c, _, _ = images.shape
        pred = self.model(images)
        _, target_index = torch.max(pred, 1)
        target_weight = self.weights[target_index]
        print()
        # classifier weight * features
        heatmaps = torch.mul(self.features, target_weight.view(n, -1, 1, 1))
        heatmaps = torch.sum(heatmaps, axis=1)
        heatmaps = resize(heatmaps, self.image_size, antialias=False, interpolation=InterpolationMode.BICUBIC)


        # heatmap normalize [0~1]
        min = (torch.min(heatmaps.view(n, -1), dim=1)[0]).view(n, 1, 1)
        max = (torch.max(heatmaps.view(n, -1), dim=1)[0]).view(n, 1, 1)

        numers = heatmaps - min
        denoms = (max - min) + 1e-4
        heatmaps = numers / denoms

        heatmaps = (heatmaps*255).to(torch.uint8)
        heatmaps = heatmaps.detach().cpu().numpy()


        # https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html#gga9a805d8262bcbe273f16be9ea2055a65afdb81862da35ea4912a75f0e8f274aeb
        colormap = cv2.COLORMAP_VIRIDIS
        heatmaps = [cv2.applyColorMap(heatmap, colormap=colormap) for heatmap in heatmaps]
        return heatmaps


# if __name__ == "__main__":
def cam(model_path, weight_path, image_size, train_loader, std, mean, device):
    # if torch.has_mps:
    #     device = torch.device("mps")

    # # data_path = "/Users/shinhyeonjun/code/class_activation_map/data/"
    # batch_size = 32
    # image_size = (256, 256)
    # mean=(.5,.5,.5)
    # std=(.5,.5,.5)
    # transforms = Compose([ToTensor(), 
    #                       Resize(size=image_size, antialias=False), 
    #                       Normalize(mean=mean, std=std)])

    # trainset = CIFAR10(root=data_path, train=True, download=True, transform=transforms, num_batchs=batch_size*2000)
    # trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)

    images, targets = next(iter(train_loader))
    images = images.to(device)
    # image = trainset[100][0][None,...]
    # target = trainset[100][1]
    # print()

    # model = torch.load('./model.pt')
    model = torch.load(model_path)
    # model_weight = torch.load('./weights_12500.pt')['model_state_dict']
    model_weight = torch.load(weight_path)['model_state_dict']
    model.load_state_dict(model_weight)
    model.to(device)
    model.eval()

    cam = Cam(model=model, image_size=image_size)
    heatmaps = cam(images)
    heatmaps = np.vstack(heatmaps)

    vis_image = images.permute(0,2,3,1).numpy()
    vis_image = np.clip(255.0 * (vis_image * std + mean), 0, 255).astype(np.uint8)
    vis_image = np.vstack(vis_image)
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

    # vis_image = torch.squeeze(image).permute(1,2,0).numpy()
    # vis_image = np.clip(255.0 * (vis_image * std + mean), 0, 255).astype(np.uint8)
    # vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

    alpha = 0.5
    output = cv2.addWeighted(vis_image, alpha, heatmaps, 1 - alpha, 0)

    output = np.hstack([vis_image, heatmaps, output])
    cv2.imwrite("cam.png", output)
    cv2.imshow("Output", output)
    cv2.waitKey(0)