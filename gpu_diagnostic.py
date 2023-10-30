from tensorflow.python.client import device_lib
import tensorflow as tf
#
# print('Tensor Flow Version: ' + tf.__version__)
#
# print('Finding Devices:')
# for dev in device_lib.list_local_devices():
#     print(dev.name)
#
# print('Listing GPUs:')
# gpu_devices = tf.config.list_physical_devices()
# tf.print(gpu_devices)
#

tf.debugging.set_log_device_placement(True)

with tf.device('/CPU:0'):
    # Create some tensors
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

    print(c)