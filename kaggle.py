import torch
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
# Login
import wandb
wandb.login()
# Necessary/extra dependencies.
import os
import gc
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
TRAIN_PATH = 'E:/장원/resized/train'
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 10

# Load image level csv file
df = pd.read_csv('E:/장원/train_image_level.csv')

# Modify values in the id column
df['id'] = df.apply(lambda row: row.id.split('_')[0], axis=1)
# Add absolute path
df['path'] = df.apply(lambda row: TRAIN_PATH+row.id+'.jpg', axis=1)
# Get image level labels
df['image_level'] = df.apply(lambda row: row.label.split(' ')[0], axis=1)

df.head(5)

# Load meta.csv file
# Original dimensions are required to scale the bounding box coordinates appropriately.
meta_df = pd.read_csv('E:/장원/meta.csv')
train_meta_df = meta_df.loc[meta_df.split == 'train']
train_meta_df = train_meta_df.drop('split', axis=1)
train_meta_df.columns = ['id', 'dim0', 'dim1']

train_meta_df.head(2)

# Merge both the dataframes
df = df.merge(train_meta_df, on='id',how="left")
df.head(2)

# Create train and validation split.
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.image_level.values)

train_df.loc[:, 'split'] = 'train'
valid_df.loc[:, 'split'] = 'valid'

df = pd.concat([train_df, valid_df]).reset_index(drop=True)

print(f'Size of dataset: {len(df)}, training images: {len(train_df)}. validation images: {len(valid_df)}')

os.makedirs('tmp/covid/images/train', exist_ok=True)
os.makedirs('tmp/covid/images/valid', exist_ok=True)

os.makedirs('tmp/covid/labels/train', exist_ok=True)
os.makedirs('tmp/covid/labels/valid', exist_ok=True)

# Move the images to relevant split folder.
for i in tqdm(range(len(df))):
    row = df.loc[i]
    if row.split == 'train':
        copyfile(row.path, f'tmp/covid/images/train/{row.id}.jpg')
    else:
        copyfile(row.path, f'tmp/covid/images/valid/{row.id}.jpg')

# Create .yaml file
import yaml

data_yaml = dict(
    train = '../covid/images/train',
    val = '../covid/images/valid',
    nc = 2,
    names = ['none', 'opacity']
)

# Note that I am creating the file in the yolov5/data/ directory.
with open('tmp/yolov5/data/data.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=True)


# Get the raw bounding box by parsing the row value of the label column.
# Ref: https://www.kaggle.com/yujiariyasu/plot-3positive-classes
def get_bbox(row):
    bboxes = []
    bbox = []
    for i, l in enumerate(row.label.split(' ')):
        if (i % 6 == 0) | (i % 6 == 1):
            continue
        bbox.append(float(l))
        if i % 6 == 5:
            bboxes.append(bbox)
            bbox = []

    return bboxes


# Scale the bounding boxes according to the size of the resized image.
def scale_bbox(row, bboxes):
    # Get scaling factor
    scale_x = IMG_SIZE / row.dim1
    scale_y = IMG_SIZE / row.dim0

    scaled_bboxes = []
    for bbox in bboxes:
        x = int(np.round(bbox[0] * scale_x, 4))
        y = int(np.round(bbox[1] * scale_y, 4))
        x1 = int(np.round(bbox[2] * (scale_x), 4))
        y1 = int(np.round(bbox[3] * scale_y, 4))

        scaled_bboxes.append([x, y, x1, y1])  # xmin, ymin, xmax, ymax

    return scaled_bboxes


# Convert the bounding boxes in YOLO format.
def get_yolo_format_bbox(img_w, img_h, bboxes):
    yolo_boxes = []
    for bbox in bboxes:
        w = bbox[2] - bbox[0]  # xmax - xmin
        h = bbox[3] - bbox[1]  # ymax - ymin
        xc = bbox[0] + int(np.round(w / 2))  # xmin + width/2
        yc = bbox[1] + int(np.round(h / 2))  # ymin + height/2

        yolo_boxes.append([xc / img_w, yc / img_h, w / img_w, h / img_h])  # x_center y_center width height

    return yolo_boxes


# Prepare the txt files for bounding box
for i in tqdm(range(len(df))):
    row = df.loc[i]
    # Get image id
    img_id = row.id
    # Get split
    split = row.split
    # Get image-level label
    label = row.image_level

    if row.split == 'train':
        file_name = f'tmp/covid/labels/train/{row.id}.txt'
    else:
        file_name = f'tmp/covid/labels/valid/{row.id}.txt'

    if label == 'opacity':
        # Get bboxes
        bboxes = get_bbox(row)
        # Scale bounding boxes
        scale_bboxes = scale_bbox(row, bboxes)
        # Format for YOLOv5
        yolo_bboxes = get_yolo_format_bbox(IMG_SIZE, IMG_SIZE, scale_bboxes)

        with open(file_name, 'w') as f:
            for bbox in yolo_bboxes:
                bbox = [1] + bbox
                bbox = [str(i) for i in bbox]
                bbox = ' '.join(bbox)
                f.write(bbox)
                f.write('\n')