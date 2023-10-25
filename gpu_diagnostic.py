from tensorflow.python.client import device_lib
import tensorflow as tf

print('Tensor Flow Version: ' + tf.__version__)

print('Finding Devices:')
for dev in device_lib.list_local_devices():
    print(dev.name)

print('Listing GPUs:')
gpu_devices = tf.config.list_physical_devices('GPU')
tf.print(gpu_devices)
