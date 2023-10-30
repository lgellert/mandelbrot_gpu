# Mandelbrot on the GPU project

Using Python and TensorFlow, compare the speed of rendering the Mandelbrot set at various resolutions on the CPU vs GPU.

See [related blog post](https://www.laurencegellert.com/2023/10/how-make-python-code-run-on-the-gpu/) for more background / details.

## Project Setup

Using `pyenv` or other virtualenv tool, create an enviroment called `mandelbrot_gpu` for this project. For example:

`pyenv virtualenv 3.9.9 mandelbrot_gpu`

Note a `.python-version` file is already setup in the repo with content `mandelbrot_gpu`.

Install python dependencies:

`pip install -r requirements.txt`

### Verify your GPU driver is working

`python gpu_diagnostic.py`

You should see the following "GPU" messages in the output:

```
Created TensorFlow device (/device:GPU with 0 MB memory) 
/device:GPU
[PhysicalDevice(name='/physical_device:GPU', device_type='GPU')]
```

### Saturate the GPU to make sure it works

To view your CPU and GPU usage, Open Activity Monitor, then Window -> GPU History (⌘4), and then Window -> CPU History (⌘3)

Run the script [in step 4 of the TensorFlow-Metal instructions](https://developer.apple.com/metal/tensorflow-plugin/) which fires up a bunch of Tensors and builds a basic machine learning model using test data.

In your GPU history window you should see it maxing out.


### Run the Mandelbrot builder script and watch your console

`python main.py`

## Results

* Date: 10/29/2023
* MacBook Pro (16-inch, 2021)
* Chip: Apple M1 Pro
* Memory: 16GB 
* macOS 12.7
* Python 3.9.9
* numpy 1.24.3
* tensorflow 2.14.0
* tensorflow-metal 1.1.0

| Device Type      | Image Size  | Time (seconds) |
|------------------|-------------|----------------|
| CPU Basic        | 500x500     | 0.484236       |
| CPU Basic        | 2500x2500   | 12.377721      |
| CPU Basic        | 5000x5000   | 47.234169      |
| TensorFlow GPU   | 500x500     | 0.372497       |
| TensorFlow GPU   | 2500x2500   | 2.682249       |
| TensorFlow GPU   | 5000x5000   | 13.176994      |
| TensorFlow GPU   | 10000x10000 | 42.316472      |
| TensorFlow GPU   | 15000x15000 | 170.987643     |
| TensorFlow CPU   | 500x500     | 0.265922       |
| TensorFlow CPU   | 2500x2500   | 2.552139       |
| TensorFlow CPU   | 5000x5000   | 12.820812      |
| TensorFlow CPU   | 10000x10000 | 46.460504      |
| TensorFlow CPU   | 15000x15000 | 328.967006     |


With the CPU Basic algorithm, I gave up after 5000x5000 because the 10000x10000 process was super low.

Between TensorFlow GPU and CPU, we can see they are about the same until 5000x5000. Then at 10000x10000 the GPU takes a small lead.
At 15000x15000 the GPU is almost twice as fast!  This shows how the marshalling of resources from the CPU to the GPU adds overhead, but
once the size of the data set is large enough the data processing aspect of the task out weights the extra cost of using the GPU.
