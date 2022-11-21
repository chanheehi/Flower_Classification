import numpy as np
import pandas as pd
import torch, os, random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pytorch_lightning as pl
from torchmetrics import F1Score, ConfusionMatrix
from CustomDataset import CustomDataset_train, CustomDataset_val, CustomDataset_test
from torch.utils.data import DataLoader
from torchvision import models
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SingleBatch:
    pred: torch.Tensor
    label: torch.Tensor
    label_name: Tuple[str]
    file_name: Tuple[str]
    file_idx: Tuple[int]

class PigModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        pl.seed_everything(42)
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier = nn.Linear(1280, 4)
        self.loss = nn.CrossEntropyLoss()
        self.f1 = F1Score(num_classes=4, average='macro')
        self.idx = -1
        self.a, self.b, self.c, self.d = torch.tensor([]), torch.tensor([]), tuple(), tuple() 

        self.train_unit = 2764    # train 개수 (전체train - val)
        self.test_unit = 563      # test 개수
        self.test_batch_unit = 64    # test 배치 개수

        self.zzamtong: List[SingleBatch] = []
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        img, label, s, img_name, img_idx = batch 
        y_f1 = label.clone()
        label = F.one_hot(label, num_classes=4).squeeze()
        label = label.to(dtype=torch.float32)
        pred = self(img).squeeze()
        loss = self.loss(pred, label)
        f1 = self.f1(pred, y_f1.squeeze())
        self.log('train_loss', loss)
        self.log('train_F1_Score', f1)

        self.zzamtong.append(
            SingleBatch(
                pred=pred,  # 예측 라벨 텐서값
                label=label,    # 정답 라벨 텐서값
                label_name=s,   # 정답 라벨 이름
                file_name=img_name, # 파일명
                file_idx=img_idx    # 파일 인덱스
            )
        )

        return {"pred": pred, "loss": loss}

    def test_step(self, batch, batch_idx):
        img, label, s, Img_name, Img_idx = batch

        # loss 계산
        y_f1 = label.clone()
        label = F.one_hot(label, num_classes=4).squeeze()
        label = label.to(dtype=torch.float32)
        pred = self(img).squeeze()
        loss = self.loss(pred, label)
        f1 = self.f1(pred, y_f1.squeeze())
        self.log('train_loss', loss)
        self.log('train_F1_Score', f1)

        self.zzamtong.append(
            SingleBatch(
                pred=pred,
                label=label,
                label_name=s,
                file_name=Img_name,
                file_idx=Img_idx
            )
        )
        
        return {"pred": pred, "loss": loss}
    
    def on_test_epoch_start(self) -> None:
        self.zzamtong = []

    def on_test_epoch_end(self) -> None:
        self.idx += 1
        
        file_name_list, label_name_list, file_idx_list= [], [], []
        
        preds = [x.pred for x in self.zzamtong]
        labels = [x.label for x in self.zzamtong]
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        
        labels = torch.argmax(labels, dim=1)
        labels = labels.view(-1, 1)
        
        
        f1 = self.f1(preds, labels) # f1 score 구함
        
        LABELS_LIST = [  # label 정의
          "sunflower",
          "rose",
          "dandelion",
          "daisy"
        ]

        # 짬통에서 리스트로 가져옴
        for x in self.zzamtong:
            file_name_list.append(x.file_name)
            label_name_list.append(x.label_name)
            file_idx_list.append(x.file_idx)

        # 1차원 리스트로 변환
        file_name_list = [data for inner_list in file_name_list for data in inner_list]
        label_name_list = [data for inner_list in label_name_list for data in inner_list]
        file_idx_list = [data for inner_list in file_idx_list for data in inner_list]

        file_la_all = []
        for i in range(labels.size(0)):
            p = torch.argmax(preds, dim=1)[i].item()
            file_la_all.append(LABELS_LIST[p])

        #submission_csv 만들기 위한 작업
        file_sub, i_num = [], 0
        f1 = str(f1).split(',')[0].split('(')[1]
        for i in range(len(file_name_list)+1):      # ===============================test의 최대 개수
            if i == 0:
                file_sub.append(['파일명', '예측라벨', '정답라벨', '채점_'+str(f1)])
                continue            
            else:
                file_sub.append([]) # 새로운 내부 배열 선언
            for j in range(1):
                file_sub[i].append(file_name_list[i_num])
                file_sub[i].append(file_la_all[i_num])
                file_sub[i].append(label_name_list[i_num])
                
                if file_sub[i][1] == file_sub[i][2]:    # 정답일 경우
                    file_sub[i].append("맞음")
                else :  # 오답일 경우
                    file_sub[i].append("틀림")
                file_sub[i].append(int(file_idx_list[i_num])) # 정렬하기 위한 인덱스 값 추가 
                i_num += 1
            
        # submission_Csv 만들기
        df = pd.DataFrame(file_sub)
        df.drop(4, axis = 1, inplace = True)    # 컬럼 삭제
        df.to_csv("./submission/A_submission"+str(self.idx)+".csv", encoding='utf-8-sig' ,index = False, header=False)    # csv 파일로 저장
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer