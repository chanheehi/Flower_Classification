import albumentations as A
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda
import numpy as np
import torch
from torchvision import datasets, transforms

def Get_train_transforms(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,    # 이동 계수
            scale_limit=0.1,    # 스케일링 인자 범위
            rotate_limit=45,    # 회전
            p=0.5,              # 변환을 적용할 확률
        ),
        A.Normalize((0.485, 0.456, 0.456), (0.229, 0.224, 0.225)),
    ])

def Get_test_transforms(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize((0.485, 0.456, 0.456), (0.229, 0.224, 0.225)),
    ])