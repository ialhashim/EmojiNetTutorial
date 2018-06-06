import os
import cv2, h5py
import numpy as np
from urllib import request

def load(resolution,count=10):
    x,y = [],[]

    for di in range(count):  
        dataset_filename = 'data/dataset'+f'{di:03}'+'.hdf5'

        # Download datasets if needed
        url_root = 'https://s3-eu-west-1.amazonaws.com/deepemoji/'
        if not os.path.isfile(dataset_filename):
            print("Downloading dataset file...")
            url = url_root + dataset_filename
            f = open(dataset_filename, 'wb')
            f.write(request.urlopen(url).read())
            f.close()
            print("Downloaded ["+url+"]")

        # Open dataset and collect all data
        print("Loading: " + dataset_filename)
        with h5py.File(dataset_filename, "r") as dataset:
            keys = []
            for key in dataset.keys(): keys.append(key.split(".")[0])
            keys = list(set(keys))

            for key in keys:
                rgb, mask = dataset.get(key+'.rgb').value, dataset.get(key+'.mask').value

                rgb = cv2.resize(rgb, dsize=(resolution, resolution))
                mask = cv2.resize(mask, dsize=(resolution, resolution))

                x.append(rgb)
                y.append(mask.reshape((mask.shape[0],mask.shape[1],1)))

    x = np.stack(x)
    y = np.stack(y)

    return x,y
