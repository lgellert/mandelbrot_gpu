import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw


# how many pixels we want, these must match (image must be square), more pixels = more run time
WIDTH = HEIGHT = 2000

# how many loops to perform when computing the mandelbrot convergence
MANDELBROT_MAX_ITERATIONS = 50

# these are the "magic numbers" that bound the mandelbrot in a cartesian grid
# this will make the mandelbrot critter centered in the output image
X_START = -2.0
X_END = 0.5
Y_START = -1.3
Y_END = 1.3

# how far to step in the mandelbrot grid for each of the output image pixels
X_STEP = (X_END - X_START) / WIDTH
Y_STEP = (Y_END - Y_START) / HEIGHT


def mandelbrot_score(c: complex, max_iterations: int) -> float:
    """
    Computes the mandelbrot score for a given complex number provided.
    Each pixel in the mandelbrot grid has a c value determined by x + 1j*y   (1j is notation for sqrt(-1))

    :param c: the complex number to test
    :param max_iterations: how many times to crunch the z value (z ** 2 + c)
    :return: float - 1 if the c value is stable, or a value 0 >= x > 1 that tells how quickly it diverged.
    """
    z = 0
    for i in range(max_iterations):
        z = z ** 2 + c
        if abs(z) > 4:
            # after it gets past abs > 4, assume it is going to infinity
            # return how soon it started spiking relative to max_iterations
            return i / max_iterations

    # c value is stable
    return 1


def display_array_as_image(data, image_title):
    img = Image.fromarray(data)
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), image_title, fill='White', font_size=24 * (WIDTH / 1000))
    img.show()


def console_progress(current, total, width=60):
    progress = current / total

    bars = int(progress * width - 1) * '-' + '>'
    spaces = int(width - len(bars)) * ' '

    # ending with a /r will replace the line
    if current < total:
        print(f'Progress: [{bars}{spaces}] {int(progress*100)}%', end='\r')
    else:
        # clear the progress line
        print('Complete' + ' ' * (width + 15))


def compute_mandelbrot_cpu():
    """
    Computes the mandelbrot set one pixel at a time using CPU and traditional coding.
    :return: array of pixels, value is divergence score
    """
    # setup a numpy array grid of pixels
    pixels = np.zeros((HEIGHT, WIDTH))

    # compute the divergence value for each pixel
    y_val = Y_START
    for y in range(HEIGHT):
        y_val = y_val + Y_STEP
        # restart the x_val counter
        x_val = X_START
        for x in range(WIDTH):
            x_val = x_val + X_STEP

            # compute the 'constant' for this pixel
            c = x_val + 1j*y_val

            # get the divergence score for this pixel
            score = mandelbrot_score(c, MANDELBROT_MAX_ITERATIONS)

            # save the score in the pixel grid
            pixels[y][x] = score

        # show progress in the console, add abs(y_val) to normalize it to zero since it is negative
        console_progress((y_val + abs(y_val)) / (Y_END + abs(y_val)), 1)

    # normalize score values to a 0 - 255 value
    pixels = pixels * 255

    return pixels


def tensor_flow_step(c_vals_, z_vals_, divergence_scores_):
    """
    The processing step for compute_mandelbrot_tensor_flow(),
    computes all pixels at once.

    :param c_vals_: array of complex values for each coordinate
    :param z_vals_: z value of each coordinate, starts at 0 and is recomputed each step
    :param divergence_scores_: the number of iterations taken before divergence for each pixel
    :return: the updated inputs
    """

    z_vals_ = z_vals_*z_vals_ + c_vals_

    # find z-values that have not diverged, and increment those elements only
    not_diverged = tf.abs(z_vals_) < 4
    divergence_scores_ = tf.add(divergence_scores_, tf.cast(not_diverged, tf.float32))

    return c_vals_, z_vals_, divergence_scores_


def compute_mandelbrot_tensor_flow(device):
    """
    Computes the mandelbrot set using TensorFlow
    :return: array of pixels, value is divergence score 0 - 255
    """

    with tf.device(device):
        # build x and y grids
        y_grid, x_grid = np.mgrid[Y_START:Y_END:Y_STEP, X_START:X_END:X_STEP]

        # compute all the constants for each pixel, and load into a tensor
        pixel_constants = x_grid + 1j*y_grid
        c_vals = tf.constant(pixel_constants.astype(np.complex64))

        # setup a tensor grid of pixel values initialized at zero
        # this will get loaded with the divergence score for each pixel
        z_vals = tf.zeros_like(c_vals)

        # store the number of iterations taken before divergence for each pixel
        divergence_scores = tf.Variable(tf.zeros_like(c_vals, tf.float32))

        # process each pixel simultaneously using tensor flow
        for n in range(MANDELBROT_MAX_ITERATIONS):
            c_vals, z_vals, divergence_scores = tensor_flow_step(c_vals, z_vals, divergence_scores)
            console_progress(n, MANDELBROT_MAX_ITERATIONS - 1)

        # normalize score values to a 0 - 255 value
        pixels_tf = np.array(divergence_scores)
        pixels_tf = 255 * pixels_tf / MANDELBROT_MAX_ITERATIONS

        return pixels_tf
