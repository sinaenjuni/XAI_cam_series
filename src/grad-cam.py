import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from datasets.cifar10 import CIFAR10
from torchsummary import summary
# from PIL import Image as pilimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as F

class GradCam(torch.nn.Module):
    def __init__(self, model, image_size):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.model.layer4.register_backward_hook(self.backward_hook)
        self.model.layer4.register_forward_hook(self.forward_hook)

    def backward_hook(self, module, input_grad, output_grad):
        self.backward_result = torch.squeeze(output_grad[0])

    def forward_hook(self, module, input, output):
        self.forward_result = torch.squeeze(output)

    def forward(self, image, target=None):
        pred = model(image)
        if target is None:
            _, target = torch.max(pred, 1)
        pred[0][target].backward(retain_graph=True)
        if self.backward_result.ndim != 3:
            self.backward_result = self.backward_result.view(-1, 1, 1)
        a_k = torch.mean(self.backward_result, dim=(1, 2), keepdim=True)         # [512, 1, 1]
        heatmap = torch.sum(a_k * self.forward_result, dim=0).detach().cpu().numpy()                  # [512, 7, 7] * [512, 1, 1]
        heatmap = cv2.resize(heatmap, self.image_size, interpolation=cv2.INTER_CUBIC)
        
        # heatmap normalize [0~1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + 1e-4
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        # out = torch.relu(out) / torch.max(out)  # 음수를 없애고, 0 ~ 1 로 scaling # [7, 7]
        # out = F.upsample_bilinear(out.unsqueeze(0).unsqueeze(0), image_size)  # 4D로 바꿈
        # return out.cpu().detach().squeeze().numpy()
        # https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html#gga9a805d8262bcbe273f16be9ea2055a65afdb81862da35ea4912a75f0e8f274aeb
        colormap = cv2.COLORMAP_VIRIDIS
        heatmap = cv2.applyColorMap(heatmap, colormap=colormap)
        return heatmap

if __name__ == "__main__":
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

    image = trainset[100][0][None,...]
    target = trainset[100][1]


    model = torch.load('./model.pt')
    model_weight = torch.load('./weights_12500.pt')['model_state_dict']
    model.load_state_dict(model_weight)
    model.eval()

    grad_cam = GradCam(model=model, image_size=image_size)
    heatmap = grad_cam(image)


    vis_image = torch.squeeze(image).permute(1,2,0).numpy()
    vis_image = np.clip(255.0 * (vis_image * std + mean), 0, 255).astype(np.uint8)
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

    alpha = 0.5
    output = cv2.addWeighted(vis_image, alpha, heatmap, 1 - alpha, 0)

    output = np.hstack([vis_image, heatmap, output])
    cv2.imshow("Output", output)
    cv2.waitKey(0)