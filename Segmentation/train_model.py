import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from Segmentation.info import *


# 反序列化原图
# @author 戴柯
# @version 1.0
def parse_img(dp):
    return tf.io.parse_tensor(dp, tf.float32)


# 反序列化掩码图
# @author 戴柯
# @version 1.0
def parse_mask(dp):
    return tf.io.parse_tensor(dp, tf.uint8)


# 从本地读取并反序列化原图和掩码图，并将原图和标签（即掩码图）打包为一个数据集
# @author 戴柯
# @version 1.0
img_ds_path = DATASET_ROOT + r'\img_ds.tfrecord'
img_ds = tf.data.TFRecordDataset(img_ds_path)
mask_ds_path = DATASET_ROOT + r'\mask_ds.tfrecord'
mask_ds = tf.data.TFRecordDataset(mask_ds_path)
img_ds = img_ds.map(parse_img)
mask_ds = mask_ds.map(parse_mask)
ds = tf.data.Dataset.zip((img_ds, mask_ds))

ds_len = 0
for dp in ds:
    ds_len += 1

# 拿到官方的oxford_iiit_pet数据集作为测试集
# @author 戴柯
# @version 1.0
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


# 对测试集进行标准化处理
# @author 戴柯
# @version 1.0
def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (MODEL_SIZE, MODEL_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (MODEL_SIZE, MODEL_SIZE))
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.uint8) - 1
    return input_image, input_mask


TRAIN_LENGTH = ds_len
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

AUTOTUNE = tf.data.experimental.AUTOTUNE


# 给反序列化后的数据手动设置shape，形式上的重新标准化，否则fit时会因为无shape属性报错
# @author 戴柯
# @version 1.0
def reprocess(dp1, dp2):
    dp1.set_shape((MODEL_SIZE, MODEL_SIZE, 3))
    dp2.set_shape((MODEL_SIZE, MODEL_SIZE, 1))
    # dp1 = tf.cast(dp1, tf.float32)
    # dp2 = tf.cast(dp1, tf.uint8)
    return dp1, dp2


train = ds.map(reprocess)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

# 从本地读取模型
# @author 戴柯
# @version 1.0
model = tf.keras.models.load_model(MODEL_PATH)

EPOCHS = EPOCHS_NUM
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[])

# 周期与精确度折线图
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

# 将模型保存到本地以实现分布式训练
# @author 戴柯
# @version 1.0
model.save(MODEL_PATH)
