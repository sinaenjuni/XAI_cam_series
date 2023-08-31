import argparse

from train import train
from cam import cam
from grad_cam import grad_cam
from grad_campp import grad_campp

def parse_args():
    parser = argparse.ArgumentParser(
            description='Simple model training funtion for Clasee Activation Map(CAM) serise')
    parser.add_argument('--mode', type=str, choices=['train', 'cam', 'grad-cam', 'grad-cam++'], 
                        help='select mode', required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print("mode:", args.mode)
    
    if args.mode == 'train':
        train()
    elif args.mode == 'cam':
        cam()
    elif args.mode == 'grad-cam':
        grad_cam()
    elif args.mode == 'grad-cam++':
        grad_campp()
        
