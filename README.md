# Kaggle-Image-Preprocessing-for-Lung-Cancer

Image Pre-processing before being fed into a 3D Convolutional Neural Network

Importing the libraries needed for the preprocessing.

```python
import numpy as np
import pandas as pd
import pydicom
import os
import matplotlib.pyplot as plt
import cv2
import math
```
Giving paths to the data directories that imput the DICOM images.

```python
dataDirectory = 'stage1/stage1/'
lungPatients = os.listdir(dataDirectory)
```
Reads the CSV values to get true values of a patient.

```python
labels = pd.read_csv('labels/labels.csv', index_col=0)
```
Setting x times y size to 50 and setting the z-dimension to 20 slices.

```python
size = 50
NoSlices = 20
def chunks(l, n):
    count = 0
    for i in range(0, len(l), n):
        if (count < NoSlices):
            yield l[i:i + n]
            count = count + 1


def mean(l):
    return sum(l) / len(l)
```
The data processing function
```python
def dataProcessing(patient, labels_df, size=50, noslices=20, visualize=False):
    label = labels_df.get_value(patient, 'cancer')
    path = dataDirectory + patient
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array), (size, size)) for each_slice in slices]

    chunk_sizes = math.floor(len(slices) / noslices)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if label == 1:
        label = np.array([0, 1])
    elif label == 0:
        label = np.array([1, 0])
    return np.array(new_slices), label


imageData = []
for num, patient in enumerate(lungPatients):
    if num % 100 == 0:
        print('Saved -', num)
    try:
        img_data, label = dataProcessing(patient, labels, size=size, noslices=NoSlices)
        imageData.append([img_data, label,patient])
    except KeyError as e:
        print('Data is unlabeled')

```
Saving the values to a .npy numpy file
```python
np.save('imageData-{}-{}-{}.npy'.format(size, size, NoSlices), imageData)
```


