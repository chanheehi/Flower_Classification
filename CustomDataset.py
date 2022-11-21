from torch.utils.data import Dataset
import numpy as np
import os, glob, cv2, torch
import csv

class CustomDataset_train(Dataset):
    def __init__(self, root_dir, transform=None, img_size=(256, 256), max_data:int = None):
        self.file_list = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)   ## root_dir의 모든 하위폴더의 모든 jpg형식 파일명 가져옴
        self.file_list = sorted(self.file_list) # 정렬
        self.transform = transform  # 정의한 트랜스폼 대입
        self.img_size = img_size  # 이미지 사이즈
        
        self.data = [["" for col in range(2)] for row in range(len(self.file_list))]
        for i, row in enumerate(self.file_list):  # 파일명과 라벨을 리스트형식으로 가져오기
          self.data[i][0] = row.split('\\')[2]  # 파일명
          self.data[i][1] = row.split('\\')[1]  # 라벨
        
        if max_data is not None:
          self.data = self.data[:max_data]
          
    def __len__(self):
        return len(self.data)  # 파이썬 구조상 반복문에서 최대 크기를 구할때 __len__함수로 들어옴
    
    def __getitem__(self, idx):   # 파이썬 구조상 a[i]와 같이 사용하면 __getitem__함수로 들어옴
        img_path = './dataset/train/'+self.data[idx][1]+"/"+self.data[idx][0]  # 파일명을 담은 리스트에서 idx번째 파일명을 img_path에 넣음
        
        img = cv2.imread(img_path)  #img_path의 파일을 img에 넣기(이미지)
        img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_AREA) 

        img = self.transform(image=img)['image']

        img = np2torch(img) # img(numpy 형식임)를 텐서형태로 만들기
        s = self.data[idx][1] # 라벨 넣음

        d = {   # label 정의
          "sunflower": 0,
          "rose": 1,
          "dandelion": 2,
          "daisy": 3
        }
        
        label = d[s]
        label = torch.LongTensor([label])
        Img_name = img_path.split('/')[4]
        Img_idx = Img_name.split('_')[0]
        return img, label, s, Img_name, Img_idx

class CustomDataset_val(Dataset):
    def __init__(self, root_dir, transform=None, img_size=(256, 256)):
        self.file_list = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)   ## root_dir의 모든 하위폴더의 모든 jpg형식 파일명 가져옴
        self.file_list = sorted(self.file_list) # 정렬
        self.transform = transform  # 정의한 트랜스폼 대입
        self.img_size = img_size  # 이미지 사이즈
        
        self.data = [["" for col in range(2)] for row in range(len(self.file_list))]
        for i, row in enumerate(self.file_list):  # 파일명과 라벨을 리스트형식으로 가져오기
          self.data[i][0] = row.split('\\')[2]  # 파일명
          self.data[i][1] = row.split('\\')[1]  # 라벨

        self.data = self.data[-500:]    # val 개수

    def __len__(self):
        return len(self.data)  # 파이썬 구조상 반복문에서 최대 크기를 구할때 __len__함수로 들어옴
    
    def __getitem__(self, idx):   # 파이썬 구조상 a[i]와 같이 사용하면 __getitem__함수로 들어옴
        img_path = './dataset/train/'+self.data[idx][1]+"/"+self.data[idx][0]  # 파일명을 담은 리스트에서 idx번째 파일명을 img_path에 넣음
        
        img = cv2.imread(img_path)  #img_path의 파일을 img에 넣기(이미지)
        img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_AREA) 

        img = self.transform(image=img)['image']

        img = np2torch(img) # img(numpy 형식임)를 텐서형태로 만들기
        s = self.data[idx][1] # 라벨 넣음

        d = {   # label 정의
          "sunflower": 0,
          "rose": 1,
          "dandelion": 2,
          "daisy": 3
        }
        
        label = d[s]
        label = torch.LongTensor([label])
        Img_name = img_path.split('/')[4]
        Img_idx = Img_name.split('_')[0]
        return img, label, Img_name, Img_idx

class CustomDataset_test(Dataset):
    def __init__(self, root_dir, transform=None, img_size=(256, 256)):
        self.file_list = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)   ## root_dir의 모든 하위폴더의 모든 jpg형식 파일명 가져옴
        self.file_list = sorted(self.file_list) # 정렬
        self.transform = transform  # 정의한 트랜스폼 대입
        self.img_size = img_size  # 이미지 사이즈

        self.data = [["" for col in range(2)] for row in range(len(self.file_list))]
        for i, row in enumerate(self.file_list):  # 파일명과 라벨을 리스트형식으로 가져오기
          if i == 0:
            pass
          self.data[i][0] = row.split('\\')[2]  # 파일명
          self.data[i][1] = row.split('\\')[1]  # 라벨
        
    def __len__(self):
        return len(self.file_list)  # 파이썬 구조상 반복문에서 최대 크기를 구할때 __len__함수로 들어옴
    
    def __getitem__(self, idx):   # 파이썬 구조상 a[i]와 같이 사용하면 __getitem__함수로 들어옴
        img_path = './dataset/test/'+self.data[idx][1]+"/"+self.data[idx][0]  # 파일명을 담은 리스트에서 idx번째 파일명을 img_path에 넣음
        
        img = cv2.imread(img_path)  #img_path의 파일을 img에 넣기(이미지)
        img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_AREA) 

        img = self.transform(image=img)['image']

        img = np2torch(img) # img(numpy 형식임)를 텐서형태로 만들기
        s = self.data[idx][1] # 라벨 넣음

        d = {   # label 정의
          "sunflower": 0,
          "rose": 1,
          "dandelion": 2,
          "daisy": 3
        }

        label = d[s]
        label = torch.LongTensor([label])
        Img_name = img_path.split('/')[4]
        Img_idx = Img_name.split('_')[0]

        return img, label, s, Img_name, Img_idx


def np2torch(x: np.ndarray):
  x = x.transpose((2, 0, 1)).astype(np.float32)
  x = x / 255.0
  x = torch.from_numpy(x)
  # print(f"np2torch: {x.min()},{x.max()}")
  return x

def torch2np(x: torch.Tensor):
  x = x.detach().cpu()
  x = x.numpy()
  x = np.clip(x * 255.0, 0, 255)
  x = x.astype(np.uint8).transpose((1, 2, 0))
  # print(f"torch2np: {x.min()},{x.max()}")
  return x
