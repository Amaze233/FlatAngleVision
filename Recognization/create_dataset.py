import tensorflow as tf
import pathlib
from Recognization.info import *
import random

AUTOTUNE = tf.data.experimental.AUTOTUNE

data_root = pathlib.Path(FLOWER_PHOTOS_PATH)
index_to_label = sorted(item.name for item in data_root.glob('*/') if item.is_dir())  # 列表
label_to_index = dict((name, index) for index, name in enumerate(index_to_label))  # 字典
all_img_paths = list(data_root.glob(r'*\*.jpg'))
random.shuffle(all_img_paths)  # 随机排序
all_img_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_img_paths]
all_img_paths = list(str(path) for path in all_img_paths)


def load_and_preprocess_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, (MODEL_SIZE, MODEL_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img


img_ds = tf.data.Dataset.from_tensor_slices(all_img_paths)
img_ds = img_ds.map(load_and_preprocess_jpg, num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_img_labels, tf.uint16))
# ds = tf.data.Dataset.zip((img_ds, label_ds))


def serialize(dp):
    return tf.io.serialize_tensor(dp)


try:
    img_ds_writer = tf.data.experimental.TFRecordWriter(DATASET_ROOT + r'\img_ds.tfrecord')
    img_ds = img_ds.map(serialize)
    img_ds_writer.write(img_ds)
    label_ds_writer = tf.data.experimental.TFRecordWriter(DATASET_ROOT + r'\label_ds.tfrecord')
    label_ds = label_ds.map(serialize)
    label_ds_writer.write(label_ds)
except BaseException as e:
    print(e)
# ds = ds.map(serialize)
# ds_writer = tf.data.experimental.TFRecordWriter(DATASET_PATH)
# ds_writer.write(ds)
