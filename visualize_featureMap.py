import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import cv2
from PIL import Image
from keras.preprocessing.image import *
from keras.models import load_model
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input
#from PSPNet import PSPNet50
from PSPNet_vgg16 import PSPNet
from FCN8s_vgg16 import FCN8
from ResCASSA_v21 import resnet_v1
from models import *
from utils.SegDataGenerator import *
from utils.metrics import *
from utils.loss_function import *
from keras.optimizers import SGD, Adam, Nadam
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
#     x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(image):
    """Takes a tensor of 3 dimensions (height, width, colors) and normalizes it's values
    to be between 0 and 1 so it's suitable for displaying as an image."""
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() )


def display_images(images, titles=None, cols=5, interpolation=None, cmap='viridis'):
    """
    images: A list of images. I can be either:
        - A list of Numpy arrays. Each array represents an image.
        - A list of lists of Numpy arrays. In this case, the images in
          the inner lists are concatentated to make one image.
    """
    images = np.transpose(images, (2,0,1))
    titles = titles or [""] * len(images)
    rows = math.ceil(len(images)*1.0 / cols)
    print (len(images), rows, cols)
    height_ratio = 1.2 * (rows/cols) * (0.5 if type(images[0]) is not np.ndarray else 1)
    plt.figure(figsize=(15, 15 * height_ratio))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.axis("off")
        # Is image a list? If so, merge them into one image.
        if type(image) is not np.ndarray:
            image = [normalize(g) for g in image]
            image = np.concatenate(image, axis=1)
        else:
            image = normalize(image)
        plt.title(title, fontsize=9)
        
        plt.imshow(image, cmap=cmap, interpolation=interpolation)
        plt.colorbar()
        i += 1
    plt.tight_layout()
    plt.show()

# Load data
train_file_path = os.path.expanduser('../segmentationDatasets/VOC2012/combined_imageset_train.txt')
val_file_path   = os.path.expanduser('../segmentationDatasets/VOC2012/combined_imageset_val_sub.txt')
data_dir        = os.path.expanduser('../segmentationDatasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
# label_dir       = os.path.expanduser('../segmentationDatasets/VOC2012/combined_annotations')
label_dir = os.path.expanduser('../segmentationDatasets/VOC2012/VOCdevkit/VOC2012/SegmentationClass')
fname = np.loadtxt(val_file_path, dtype='str')
findex = np.where(fname == '2007_009096')[0][0]#2008_005890, 2011_001591
print (os.path.join(data_dir,fname[findex]+'.jpg'))
img = cv2.imread(os.path.join(data_dir,fname[findex]+'.jpg'))
cv2.imwrite('pred.png', img)
label = cv2.imread(os.path.join(label_dir,fname[findex]+'.png'))

data_suffix='.jpg'
label_suffix='.png'
classes = 21
target_size = (320,320)
ignore_label=255
label_cval=255
loss_shape = None
batch_size=1
train_datagen = SegDataGenerator(featurewise_center=False,
                                     featurewise_std_normalization=False,
                                    # zoom_range=[0.5, 2.0],
                                     zoom_maintain_shape=True,
                                     crop_mode='none',
                                     crop_size=target_size,
                                     # pad_size=(505, 505),
                                     rotation_range=0.,
                                     shear_range=0,
                                     horizontal_flip=False,
                                    # channel_shift_range=20.,
                                     fill_mode='constant',
                                     label_cval=label_cval)

train_generator = train_datagen.flow_from_directory(
                    file_path=val_file_path,
                    data_dir=data_dir, data_suffix=data_suffix,
                    label_dir=label_dir, label_suffix=label_suffix,
                    classes=classes,
                    target_size=target_size, color_mode='rgb',
                    batch_size=batch_size, shuffle=False,
                    loss_shape=loss_shape,
                    ignore_label=ignore_label,
                    # save_to_dir='Images/'
                    )
x , y = train_generator._get_batches_of_transformed_samples([findex])
print(x.shape)

# ######

labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
index = labels.index('person')
layer_index = -17#-46#-34
input_img_data = x
checkpoint_path = os.path.join('./Models/ResCBAM', 'checkpoint_weights.hdf5')
model = resnet_v1(input_shape=(320, 320, 3), num_classes=21, attention_module='hough_glass')

# model.summary()
lr_base = 1e-5
loss_fn = softmax_sparse_crossentropy_ignoring_last_label
optimizer = Adam(lr=lr_base, beta_1 = 0.825, beta_2 = 0.99685)
metrics=[sparse_accuracy_ignoring_last_label]

# model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
model.compile(loss=[loss_fn, loss_fn,loss_fn,loss_fn,loss_fn,loss_fn],
                  optimizer=optimizer,
                  metrics=metrics)
model.load_weights(checkpoint_path, by_name=False)
# intermediate_layer_model = Model(inputs=model.input,
#                                  outputs=model.layers[layer_index].output)

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('fuse').output) #hough_glass_lvl1, ssa_mul3, lvl4_mul

intermediate_output = intermediate_layer_model.predict(input_img_data)
res = intermediate_output[0]
print (res.shape)

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224, 192]
pred = np.argmax(np.squeeze(res), axis=-1).astype(np.uint8)
print (pred[150, :])
# print (res[20,20])
pred = Image.fromarray(pred, mode='P')
# label = Image.fromarray(label, mode='P')
pred.putpalette(palette)
# label.putpalette(palette)
pred.save('test.png')
# label.save('label.png')
cv2.imwrite('label.png', label)
pred = np.array(pred)
fig=plt.figure(figsize=(50, 50))
fig.add_subplot(1, 3, 1)
plt.imshow(img)
fig.add_subplot(1, 3, 2)
plt.imshow(label)
fig.add_subplot(1, 3, 3)
plt.imshow(pred)
plt.show()
