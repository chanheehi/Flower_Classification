import os, torch, random
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import callbacks, seed_everything
import torch.nn as nn
from torchmetrics import F1Score
import torchmetrics
import numpy as np
# 다른 .py 파일 import
from CustomDataset import CustomDataset_train, CustomDataset_val, CustomDataset_test
from Nets import PigNetV1
from Train import PigModule
from Augmentation import get_train_transforms

if __name__ == "__main__":
    # 시드 및 GPU 설정
    
    seed = 42 # seed 값 설정
    random.seed(seed) # 파이썬 난수 생성기 
    os.environ['PYTHONHASHSEED'] = str(seed) # 해시 시크릿값 고정
    seed_everything(42, workers=True)   # 파이토치 시드 고정
    avail_gpus = max(1, torch.cuda.device_count())  # 사용 가능한 최대 gpu 개수

    # 경로 설정
    root_dir = "./dataset/" # 기본 경로
    train_dir, test_dir = [os.path.join(root_dir, s) for s in ["train", "test"]]    #root_dir의 하위폴더에서 train폴더면 train_dir로, test폴더면 test_dir로

    trainer = pl.Trainer(
        max_epochs=5,    # 에폭
        deterministic=True, # 재현성을 보장
        gpus = avail_gpus,  #사용가능한 모든 GPU
        callbacks=[
            # F1Callback(r),
            callbacks.ModelCheckpoint(
                dirpath="{checkpoints",  # 저장 위치
                filename='{Epoch}_{train_loss}.3f_best',    # 파일명
                save_top_k=1,   # 베스트 모델 상위 n개까지 저장
                monitor="train_loss",   # loss가 기준
                mode="min",     # loss는 낮을수록 좋은것이니 min
            ),
        ],
        logger=pl.loggers.TensorBoardLogger("logs"),
        log_every_n_steps=1,
        )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # GPU 병렬 사용 가능한지 여부
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(PigModule)
        model = model.to(device)
        print("Let's use", torch.cuda.device_count(), "GPUs!")  # 사용 가능한 GPU 출력
    elif torch.cuda.device_count() <= 1:
        model = PigModule().to(device)
        print("Let's use", torch.cuda.device_count(), "GPUs!")  # 사용 가능한 GPU 출력

    model.load_state_dict(torch.load('model.pth'), strict=False)

    # 학습 준비
    test_dir = os.path.join(root_dir, 'test')
    train_transform = get_train_transforms()

    train_set = CustomDataset_train(train_dir, train_transform)
    val_set = CustomDataset_val(train_dir, train_transform)
    test_set = CustomDataset_test(test_dir)
    train_loader = DataLoader(train_set, batch_size = 4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size = 4, shuffle=False)
    test_loader = DataLoader(test_set, batch_size = 4,shuffle=False)

    # 학습
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="./checkpoints/1.ckpt")   # 학습된 모델중 최적의 모델로 test

        # 모델 저장
    torch.save(model.state_dict(), './model.pth_best')