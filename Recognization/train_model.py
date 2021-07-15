import tensorflow as tf
from Recognization.info import *
import matplotlib.pyplot as plt

model = tf.keras.models.load_model(MODEL_PATH)


def parse_img(dp):
    dp = tf.io.parse_tensor(dp, tf.float32)
    dp.set_shape((MODEL_SIZE, MODEL_SIZE, 3))
    return dp


def parse_label(dp):
    return tf.io.parse_tensor(dp, tf.uint16)


img_ds = tf.data.TFRecordDataset(DATASET_ROOT + r'\img_ds.tfrecord')
label_ds = tf.data.TFRecordDataset(DATASET_ROOT + r'\label_ds.tfrecord')
img_ds = img_ds.map(parse_img)
label_ds = label_ds.map(parse_label)
ds = tf.data.Dataset.zip((img_ds, label_ds))

ds_len = 0
for dp in ds:
    ds_len += 1

ds = ds.batch(BATCH_SIZE)
# vds = ds
# vds_len = ds_len

TRAIN_LENGTH = ds_len
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

# VALIDATION_LENGTH = vds_len
# VAL_SUBSPLITS = 5
# VALIDATION_STEPS = VALIDATION_LENGTH // BATCH_SIZE // VAL_SUBSPLITS

history = model.fit(ds, epochs=EPOCHS_NUM, shuffle=1, steps_per_epoch=STEPS_PER_EPOCH)
# history = model.fit(ds, epochs=EPOCHS_NUM, shuffle=1, steps_per_epoch=STEPS_PER_EPOCH,
#                     validation_steps=VALIDATION_STEPS, validation_data=vds)


# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()

model.save(MODEL_PATH)
