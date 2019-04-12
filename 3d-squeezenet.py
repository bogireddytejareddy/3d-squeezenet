import keras.backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Flatten, Dropout, Concatenate, Activation, Dense
from keras.layers import Convolution3D, MaxPooling3D, AveragePooling3D
from keras.layers import GlobalMaxPooling3D, GlobalAveragePooling3D

from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file

K.set_image_dim_ordering('th')

input_shape = (1, 227, 227, 96)
input_tensor = None

if input_tensor is None:
        img_input = Input(shape=input_shape)
else:
        if not K.is_keras_tensor(input_tensor):
            	img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
		img_input = input_tensor

print img_input

def firemodule(x, filters, name="firemodule"):
	squeeze_filter, expand_filter1, expand_filter2 = filters
	squeeze = Convolution3D(squeeze_filter, (1, 1, 1), activation='relu', padding='same', name=name + "/squeeze1x1x1")(x)
	expand1 = Convolution3D(expand_filter1, (1, 1, 1), activation='relu', padding='same', name=name + "/expand1x1x1")(squeeze)
	expand2 = Convolution3D(expand_filter2, (3, 3, 3), activation='relu', padding='same', name=name + "/expand3x3x3")(squeeze)
	x = Concatenate(axis=-1, name=name)([expand1, expand2])
	return x

x = Convolution3D(64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same", activation="relu", name='conv1')(img_input)
x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='maxpool1', padding="valid")(x)

x = firemodule(x, (16, 64, 64), name="fire2")
x = firemodule(x, (16, 64, 64), name="fire3")

x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='maxpool3', padding="valid")(x)
x = firemodule(x, (32, 128, 128), name="fire4")
x = firemodule(x, (32, 128, 128), name="fire5")
x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='maxpool5', padding="valid")(x)
x = firemodule(x, (48, 192, 192), name="fire6")
x = firemodule(x, (48, 192, 192), name="fire7")
x = firemodule(x, (64, 256, 256), name="fire8")
x = firemodule(x, (64, 256, 256), name="fire9")

x = GlobalMaxPooling3D(name="maxpool10")(x)
x = Dense(3, init='normal')(x)
x = Activation('softmax')(x)

model = Model(img_input, x, name="squeezenet")
model.summary()

