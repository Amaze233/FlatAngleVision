import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
from Recognization.info import *

model = tf.keras.models.load_model(MODEL_PATH)

file_root = WORK_SPACE


def load_and_decode_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (MODEL_SIZE, MODEL_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img


file_paths = list(pathlib.Path(file_root).glob('*.jpg')) + list(pathlib.Path(file_root).glob('*.png'))
file_paths = list(str(path) for path in file_paths)
file_ds = tf.data.Dataset.from_tensor_slices(file_paths)
file_ds = file_ds.map(load_and_decode_jpg)
file_ds = file_ds.batch(1)

from Recognization.create_dataset import index_to_label

for file in file_ds:
    plt.imshow(tf.keras.preprocessing.image.array_to_img(file[0]))
    plt.show()
    result = model.predict(file)
    buf = 0
    index = 0
    index_buf = -1
    for num in result[0]:
        if num > buf:
            buf = num
            index_buf = index
        index += 1
    print(index_to_label[index_buf])
