import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import os
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore", category=UserWarning)

torch.manual_seed(1)

class MyMnistDataset(Dataset):
    def __init__(self, filePath):

        self.myMnistPath = filePath
        self.imagesData = []

        self.loadImageData()

    # 读取手写图片数据，并将图片数据和对应的标签组合在一起
    def loadImageData(self):
        imagesFolderPath = os.path.join(self.myMnistPath, 'input_images')
        imageFiles = os.listdir(imagesFolderPath)

        for imageName in imageFiles:
            imagePath = os.path.join(imagesFolderPath, imageName)
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            self.imagesData.append(image)

        self.imagesData = torch.Tensor(self.imagesData).unsqueeze(1)

    # 重写魔法函数
    def __getitem__(self, index):
        return self.imagesData[index]

    # 重写魔法函数
    def __len__(self):
        return len(self.imagesData)

#载入自己的数据集
dataset = MyMnistDataset('../MyMnistTest')
test_loader = DataLoader(dataset=dataset, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # (1,28,28)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                      stride=1, padding=2),  # (16,28,28)
            # 想要conv2d卷积出来的图片尺寸没有变化, padding=(kernel_size-1)/2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # (16,14,14)
        )
        self.conv2 = nn.Sequential(  # (16,14,14)
            nn.Conv2d(16, 32, 5, 1, 2),  # (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)  # (32,7,7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 将（batch，32,7,7）展平为（batch，32*7*7）
        output = self.out(x)
        return output


#载入训练好的模型
model = CNN()
model.load_state_dict(torch.load("CNNMnist.pth"))

def test():
    print("predicted")
    with torch.no_grad():
        for data in test_loader:
            images = data
            ouputs = model(images)
            _, predicted = torch.max(ouputs.data, dim=1)

            print("{}".format(predicted.data.item()))



if __name__ == '__main__':
    test()
