import sys
import os

# path = 'C:/Users/jwein/Desktop/python/'
path = 'C:/Users/poohl/PycharmProjects/kaggleCovid19'
sys.path.append(path)
os.chdir(path)

import numpy as np
import pandas as pd
import os
import pydicom
import glob
import tensorflow as tf
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
from tqdm import tqdm
import torch
from torch.nn import Sequential, Module, Conv2d, ReLU, BatchNorm2d, MaxPool2d, Linear, CrossEntropyLoss, MSELoss
from torch.optim import Adam
import warnings
warnings.simplefilter('ignore')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# preparing train set
train_image = pd.read_csv("train_image_level.csv")
train_study = pd.read_csv("train_study_level.csv")

TRAIN_DIR = "../train/"
train_study['StudyInstanceUID'] = train_study['id'].apply(lambda x: x.replace('_study', ''))
train = train_image.merge(train_study, on='StudyInstanceUID')

# Make a path folder(train set)
paths = []
for instance_id in train['StudyInstanceUID']:
    paths.append(glob.glob(os.path.join(TRAIN_DIR, instance_id +"/*/*"))[0])

train['path'] = paths

train = train.drop(['id_x', 'id_y'], axis=1)

# load submission
# sub_df = pd.read_csv('../kaggle/input/siim-covid19-detection/sample_submission.csv')
sub_df = pd.read_csv('sample_submission.csv')
study_df = sub_df.loc[sub_df.id.str.contains('_study')]
image_df = sub_df.loc[sub_df.id.str.contains('_image')]

def dicom2array(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.dcmread(path)
    dicom.PhotometricInterpretation = 'YBR_FULL'
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

def resize_xray(array, size, keep_ratio=False, resample=Image.LANCZOS):
    im = Image.fromarray(array)

    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)

    return im

set_n = 200
X_train = []
y_train = train[['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']].iloc[:set_n]

error_list = []
for i in tqdm(range(set_n)):
    try:
        X_train.append(np.array(resize_xray(dicom2array(train['path'][i]), 32)))
    except RuntimeError:
        error_list.append(i)
        y_train = y_train.drop([y_train.index[i]])
        pass

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)

# validation set
X_valid = []
valid_n = np.random.randint(train.shape[0], size=10)
y_valid = train[['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']].iloc[valid_n]


for i in tqdm(range(len(valid_n))):
    try:
        X_valid.append(np.array(resize_xray(dicom2array(train['path'][i]), 32)))
    except RuntimeError:
        y_valid = y_valid.drop([y_valid.index[i]])
        pass
X_valid, y_valid = np.array(X_valid), np.array(y_valid)
X_valid = X_valid.reshape(X_valid.shape[0], 1, X_valid.shape[1], X_valid.shape[2])
X_valid, y_valid = torch.from_numpy(X_valid), torch.from_numpy(y_valid)

class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = Sequential(
            Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            ReLU(True),
            MaxPool2d(kernel_size=2, stride=2)
        ) # L1 ImgIn shape=(?, 1, 32, 32), Conv->(?, 32, 32, 32), Pool->(?, 32, 16, 16)
        self.layer2 = Sequential(
            Conv2d(32, 64, 3, 1, 1),
            ReLU(True),
            MaxPool2d(2, 2)
        ) # L2 ImgIn shape=(?, 32, 16, 16), Conv->(?, 64, 16, 16), Pool-> (?, 64, 8, 8)
        self.layer3 = Sequential(
            Conv2d(64, 128, 3, 1, 1),
            ReLU(True),
            MaxPool2d(2, 2)
        ) # L3 ImgIn shape=(?, 64, 8, 8), Conv->(?, 128, 8, 8), Pool->(?, 128, 4, 4)
        self.fc1 = Linear(4 * 4 * 128, 128, bias=True)
        self.fc2 = Linear(128, 4, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(-1, 128*4*4)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = CNN().to(DEVICE)
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.01)
# defining the loss function
criterion = MSELoss().to(DEVICE)

for epoch in range(500):
    running_loss = 0.0
    for index in range(X_train.shape[0]):
        X = X_train[index].reshape(1, 1, X_train.shape[2], X_train.shape[3]).float().to(DEVICE)
        Y = y_train[index].reshape(-1, y_train.shape[1]).to(DEVICE)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if index % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

print('Finished Training')