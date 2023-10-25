import time
import os

# disable TensorFlow notices
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mandelbrot import compute_mandelbrot_cpu, display_array_as_image, WIDTH, HEIGHT, compute_mandelbrot_tensor_flow

print('Building Mandelbrot using CPU - Resolution %dx%d' % (WIDTH, HEIGHT))
start = time.time()
pixels_cpu = compute_mandelbrot_cpu()
end = time.time()
total_time = round(end - start, 6)

# report the time it took and summary data
print('Pixel array computed in', total_time, 'seconds')
print('Pixels computed', len(pixels_cpu), 'x', len(pixels_cpu[0]))

# show the CPU image
display_array_as_image(pixels_cpu, 'Mandelbrot CPU %dx%d in %s seconds' % (WIDTH, HEIGHT, total_time))


print('------------------------------------')
print('Building Mandelbrot using TensorFlow (CPU Device) - Resolution %dx%d' % (WIDTH, HEIGHT))

start = time.time()
pixels_tf = compute_mandelbrot_tensor_flow('/CPU:0')
end = time.time()
total_time = round(end - start, 6)


# report the time it took and summary data
print('Pixel array computed in', total_time, 'seconds')
print('Pixels computed', len(pixels_tf), 'x', len(pixels_tf[0]))

# show the image
display_array_as_image(pixels_tf, 'Mandelbrot TensorFlow (CPU Device) %dx%d in %s seconds' % (WIDTH, HEIGHT, total_time))


print('------------------------------------')
print('Building Mandelbrot using TensorFlow (GPU Device) - Resolution %dx%d' % (WIDTH, HEIGHT))

start = time.time()
pixels_tf = compute_mandelbrot_tensor_flow('/GPU:0')
end = time.time()
total_time = round(end - start, 6)

# report the time it took and summary data
print('Pixel array computed in', total_time, 'seconds')
print('Pixels computed', len(pixels_tf), 'x', len(pixels_tf[0]))

# show the image
display_array_as_image(pixels_tf, 'Mandelbrot TensorFlow (GPU Device) %dx%d in %s seconds' % (WIDTH, HEIGHT, total_time))