import time
import os

# disable TensorFlow notices
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mandelbrot import MandelbrotCPUBasic, MandelbrotTensorFlow


print('------------------------------------')
print('Building Mandelbrot using basic CPU logic')
print('| Device Type | Size        | Time (seconds) |')
for size in [500, 2500, 5000]:

    mb_logic = MandelbrotCPUBasic(size, 50)
    start = time.time()
    pixels_cpu = mb_logic.compute()
    end = time.time()
    total_time = round(end - start, 6)

    # report the time it took and summary data
    print('| CPU Basic | %dx%d | %s |' % (size, size, total_time))

print('------------------------------------')
print('Building Mandelbrot using TensorFlow:')
print('| Device Type | Size        | Time (seconds) |')

for device in ['/GPU:0', '/CPU:0']:
    for size in [500, 2500, 5000, 10000, 15000]:

        mb_logic = MandelbrotTensorFlow(size, 250)
        start = time.time()
        pixels_tf = mb_logic.compute(device=device)
        end = time.time()
        total_time = round(end - start, 6)

        # report the time it took and summary data
        print('| %s | %dx%d | %s |' % ('TensorFlow ' + device, size, size, total_time))


# show the last image
mb_logic.display_array_as_image(pixels_tf, 'Mandelbrot TensorFlow (%s) %dx%d in %s seconds' % (device, size, size, total_time))
