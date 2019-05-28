**Transfer Model: 用一个已经训练好的模型，把最后的prediction layer去掉，然后训练自己的模型。**

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning.exercise_5 import *
print("Setup Complete")

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = 224

# Specify the values for all arguments to data_generator_with_aug.
# 底下加注释的是data augmentation
data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                              horizontal_flip = True, # 水平旋转
                                              width_shift_range = 0.1, # shift
                                              height_shift_range = 0.1) # shift
            
data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator_with_aug.flow_from_directory(
        directory = '../input/dogs-gone-sideways/train',
        target_size=(image_size, image_size),
        batch_size=12,
        class_mode='categorical')

# Specify which type of ImageDataGenerator above is to load in validation data
validation_generator = data_generator_no_aug.flow_from_directory(
        directory = '../input/dogs-gone-sideways/val',
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator,
        epochs = 3, #循环多少轮
        steps_per_epoch=19, # 每一轮的步数，上面的batch size是每一步进行的计算的图片数
        validation_data=validation_generator)
```

**From Scratch**

```python
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw):
    # 数据的第一列是label，作为结果来存进y
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    # 数据的其他列作为input存为x
    x = raw[:,1:]
    num_images = raw.shape[0]# 这是行数，shape[1]是列数
    # 这里的shape要和后面的input_shape相一致。
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    # 这一步似乎是normalization？？？
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data)

# Your code below
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D

second_fashion_model = Sequential()

# 加入一个Convotional layer， 20 个filter，kernel size是3x3
second_fashion_model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
# 后面的Conv2D layer不用写input shape了。
second_fashion_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
second_fashion_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
# 把前面的layer flatten一下，参考官方文档
second_fashion_model.add(Flatten())
# 加一个Dense layer
second_fashion_model.add(Dense(100, activation='relu'))
second_fashion_model.add(Dense(num_classes, activation='softmax'))
# compile这个model
second_fashion_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
# 训练这个model，用之前读出来的x y
second_fashion_model.fit(x, y,
          batch_size=100,
          epochs=4,
          validation_split = 0.2)
```

**Dropout and strides**

```python
# 会drop 50%的units
model.add(Dropout(0.5))
# 步长为2
model.add(Conv2D(30, kernel_size=(3, 3), strides=2, activation='relu'))
```

