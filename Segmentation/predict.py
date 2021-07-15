import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import cv2
from info import *

model = tf.keras.models.load_model(MODEL_PATH)

file_root = UN_PRODUCED_FILE_PTAH
save_root = PRODUCED_FILE_PATH


def load_and_decode_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    return img


# 读取before目录下所有jpg png格式文件并转为张量数据集
# @author 戴柯
# @version 1.0
file_paths_jpg = list(pathlib.Path(file_root).glob('*.jpg'))
file_paths_png = list(pathlib.Path(file_root).glob('*.png'))
file_paths = file_paths_jpg + file_paths_png
file_paths = [str(path) for path in file_paths]
file_ds = tf.data.Dataset.from_tensor_slices(file_paths)
file_ds = file_ds.map(load_and_decode_img)
# file_ds = file_ds.batch(len(file_paths))
file_ds = file_ds.batch(1)


# 将predict返回的标签掩码图张量译码为012的格式
# @author 戴柯
# @version 1.0
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    pred_mask = pred_mask[0]
    # pred_mask = tf.cast(pred_mask, tf.uint32)
    return pred_mask


# 使用模型对图片文件进行预测，返回译码后的掩码图
# @author 戴柯
# @version 1.0
def process(file):
    img = tf.image.resize(file, (MODULE_SIZE, MODULE_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    mask = create_mask(model.predict(img))
    mask = tf.image.resize(mask, (file.shape[1], file.shape[2]))
    plt.imshow(tf.keras.preprocessing.image.array_to_img(file[0]))
    plt.show()
    plt.imshow(tf.keras.preprocessing.image.array_to_img(mask))
    plt.show()
    return mask


mask_list = [process(file) for file in file_ds]

img_list = [cv2.imread(file) for file in file_paths]

# 根据掩码图对原图进行处理，并将结果保存在after文件夹下
# @author 戴柯
# @version 1.0
for i in range(len(mask_list)):
    img_list[i] = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2BGRA)
    mask_list[i] = np.repeat(mask_list[i], 4, 2)
    img_list[i] = np.multiply(img_list[i], mask_list[i])
    # img = img.astype(np.int)
    # img_list[i] /= 255.0
    cv2.imwrite(r'{0}\{1}.png'.format(save_root, pathlib.Path(file_paths[i]).stem), img_list[i])