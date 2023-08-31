import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from datasets.cifar10 import CIFAR10
from torchsummary import summary
# from PIL import Image as pilimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision.transforms.functional import resize, InterpolationMode

class GradCampp(torch.nn.Module):
    def __init__(self, model, image_size):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.model.layer4.register_backward_hook(self.backward_hook)
        self.model.layer4.register_forward_hook(self.forward_hook)
        self.forward_result = []
        self.backward_result = []
        
    def backward_hook(self, module, input_grad, output_grad):
        self.backward_result += [output_grad[0]]

    def forward_hook(self, module, input, output):
        self.forward_result = output

    def forward(self, images, target=None):
        n, c, _, _ = images.shape

        pred = self.model(images)
        max_logits, indices = torch.max(pred, 1)
        for max_logit in max_logits:
            self.model.zero_grad()
            max_logit.backward(retain_graph=True)

        self.backward_result = torch.stack([self.backward_result[idx][idx] for idx in range(n)])
        print()
        # calculate alpha
        numerator = self.backward_result.pow(2)
        denominator = 2 * self.backward_result.pow(2)
        ag = self.forward_result * self.backward_result.pow(3)
        denominator += torch.sum(ag, dim=(2,3), keepdim=True)
        # denominator += ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
        denominator = torch.where(
            denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = numerator / (denominator + 1e-7)

        # relu_grad = F.relu(pred[0, indices].exp() * self.backward_result)
        relu_grad = F.relu(max_logits.exp().view(-1, 1, 1, 1) * self.backward_result)
        # weights = (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)
        weights = torch.sum((alpha * relu_grad), dim=(2,3), keepdim=True)

        heatmaps = torch.sum(weights * self.forward_result, dim=1)

        # if self.backward_result.ndim != 3:
            # self.backward_result = self.backward_result.view(-1, 1, 1)
        # a_k = torch.mean(self.backward_result, dim=(1, 2), keepdim=True)         # [512, 1, 1]
        # heatmap = torch.sum(a_k * self.forward_result, dim=0).detach().cpu().numpy()                  # [512, 7, 7] * [512, 1, 1]
        # heatmap = cv2.resize(heatmap, self.image_size, interpolation=cv2.INTER_CUBIC)
        heatmaps = resize(heatmaps, self.image_size, antialias=False, interpolation=InterpolationMode.BICUBIC)
        
        # heatmap normalize [0~1]
        min = (torch.min(heatmaps.view(n, -1), dim=1)[0]).view(n, 1, 1)
        max = (torch.max(heatmaps.view(n, -1), dim=1)[0]).view(n, 1, 1)

        numers = heatmaps - min
        denoms = (max - min) + 1e-4
        heatmaps = numers / denoms

        heatmaps = (heatmaps*255).to(torch.uint8)
        heatmaps = heatmaps.detach().cpu().numpy()

        # out = torch.relu(out) / torch.max(out)  # 음수를 없애고, 0 ~ 1 로 scaling # [7, 7]
        # out = F.upsample_bilinear(out.unsqueeze(0).unsqueeze(0), image_size)  # 4D로 바꿈
        # return out.cpu().detach().squeeze().numpy()
        # https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html#gga9a805d8262bcbe273f16be9ea2055a65afdb81862da35ea4912a75f0e8f274aeb
        colormap = cv2.COLORMAP_VIRIDIS
        heatmaps = [cv2.applyColorMap(heatmap, colormap=colormap) for heatmap in heatmaps]
        return heatmaps

# if __name__ == "__main__":
def grad_campp():
    if torch.has_mps:
        device = torch.device("mps")

    data_path = "/Users/shinhyeonjun/code/class_activation_map/data/"
    batch_size = 32
    image_size = (256, 256)
    mean=(.5,.5,.5)
    std=(.5,.5,.5)
    transforms = Compose([ToTensor(), 
                          Resize(size=image_size, antialias=False), 
                          Normalize(mean=mean, std=std)])

    trainset = CIFAR10(root=data_path, train=True, download=True, transform=transforms, num_batchs=batch_size*2000)
    trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)

    images, targets = next(iter(trainloader))

    # image = trainset[100][0][None,...]
    # target = trainset[100][1]
    # print()

    model = torch.load('./model.pt')
    model_weight = torch.load('./weights_12500.pt')['model_state_dict']
    model.load_state_dict(model_weight)
    model.eval()

    cam = GradCampp(model=model, image_size=image_size)
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
    cv2.imshow("Output", output)
    cv2.waitKey(0)