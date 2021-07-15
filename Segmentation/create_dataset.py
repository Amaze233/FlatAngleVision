import tensorflow as tf
import pathlib
from Segmentation.info import *

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_img(img_path):
    img = tf.io.read_file(img_path)
    return img


# 从winpath路径读取原图文件，并将其大小设为模型所需大小，之后通过转为tf.float32再除以255.0进行标准化
# @author 戴柯
# @version 1.0
def load_and_preprocess_jpg(img_path, size=MODEL_SIZE):
    img = load_img(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (size, size))
    img = tf.cast(img, tf.float32) / 255.0
    return img


# 从路径读取掩码图文件，并将其设为模型所需大小，-1将123转为012进行标准化，再转为tf.uint8缩减大小以便保存
# @author 戴柯
# @version 1.0
def load_and_preprocess_mask(mask_path, size=MODEL_SIZE):
    mask = load_img(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (size, size))
    mask -= 1
    mask = tf.cast(mask, tf.uint8)  # may be the reason of missing of data ! ! !
    return mask


img_root = pathlib.Path(IMG_PATH)
all_img_paths = list(img_root.glob('*.jpg'))
all_img_paths += list(img_root.glob('*.png'))
mask_root = pathlib.Path(MASK_PATH)
all_mask_paths = list(mask_root.glob('*.png'))
all_mask_paths += list(mask_root.glob('*.jpg'))

all_img_paths_able = []
all_mask_paths_able = []
num = 0
# 匹配同名原图和掩码图，加入_able列表中
# @author 戴柯
# @version 1.0
print('match img&mask:')
for img_path in all_img_paths:
    for mask_path in all_mask_paths:
        if img_path.stem == mask_path.stem:
            all_img_paths_able.append(img_path)
            all_mask_paths_able.append(mask_path)
            num += 1
            print(num)
print('matched')

all_img_paths_able = [str(path) for path in all_img_paths_able]
all_mask_paths_able = [str(path) for path in all_mask_paths_able]

# 生成数据集
# @author 戴柯
# @version 1.0
img_path_ds = tf.data.Dataset.from_tensor_slices(all_img_paths_able)
mask_path_ds = tf.data.Dataset.from_tensor_slices(all_mask_paths_able)
img_ds = img_path_ds.map(load_and_preprocess_jpg, num_parallel_calls=AUTOTUNE)
mask_ds = mask_path_ds.map(load_and_preprocess_mask, num_parallel_calls=AUTOTUNE)
# ds = tf.data.Dataset.zip((img_ds, mask_ds))
# print(ds)


# 序列化函数，供map调用
# @author 戴柯
# @version 1.0
def serialize(dp):
    return tf.io.serialize_tensor(dp)


# 分别保存原图和掩码图数据集为tfreorder格式于voc2012文件夹下
# @author 戴柯
# @version 1.0
img_ds_path = DATASET_ROOT + r'\img_ds.tfrecord'
img_ds_writer = tf.data.experimental.TFRecordWriter(img_ds_path)
img_ds_serialized = img_ds.map(serialize)
img_ds_writer.write(img_ds_serialized)
mask_ds_path = DATASET_ROOT + r'\mask_ds.tfrecord'
mask_ds_writer = tf.data.experimental.TFRecordWriter(mask_ds_path)
mask_ds_serialized = mask_ds.map(serialize)
mask_ds_writer.write(mask_ds_serialized)
