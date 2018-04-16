'''
This script goes along my blog post:
"Extending Keras' ImageDataGenerator to Support Random Cropping" (https://jkjung-avt.github.io/keras-image-cropping/)
'''


from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing import image
import os
import sys
import glob
import argparse
import numpy as np


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    return args


def get_files(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*'))
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    files = [f for f in files if f.endswith('JPG') or f.endswith('jpg')]

    if not len(files):
        sys.exit('No images found by the given path!')

    return files


def center_crop(img, crop_size):
    height, width = img.shape[0], img.shape[1]
    dy, dx = crop_size
    x = (width - dx + 1) // 2
    y = (height - dy + 1) // 2
    return img[y:(y+dy), x:(x+dx), :]


if __name__ == '__main__':
    args = parse_args()
    files = get_files(args.path)
    cls_list = ['cats', 'dogs']

    # load the trained model
    net = load_model('model-resnet50-final.h5')

    # loop through all files and make predictions
    for f in files:
        img = image.load_img(f, target_size=(256,256))
        if img is None:
            continue
        x = image.img_to_array(img)
        x = preprocess_input(x)
        x = center_crop(x, (224,224))
        x = np.expand_dims(x, axis=0)
        pred = net.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        print(f)
        for i in top_inds:
            print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
