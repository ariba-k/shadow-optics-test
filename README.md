# IPA LED Test ML Model

Machine learning model for optical alignment prediction during the IPA LED test.

## Installation

See the [TensorFlow documentation](https://www.tensorflow.org/install/pip#hardware_requirements) for details. These instructions are based on a Windows PC with NVIDIA GPU.

* Install Miniconda ([conda.io](https://docs.conda.io/en/latest/miniconda.html))
* Download this project (e.g. using `git clone`)
* Open the project
  * No IDE: Open "Anaconda Prompt" or "Anaconda Prompt (PowerShell)" and `cd` to the project directory
  * PyCharm: File > Open...
  * VSCode: File > Open Folder...
* Run the following commands:
  * `conda env create --file environment.yaml`
  * `conda activate tf`
  * `pip install -r requirements.txt`

### GPU Support

Before installing TensorFlow:

* Install the latest GPU driver ([nvidia.com](https://www.nvidia.com/Download/index.aspx))
* Install CUDA toolkit ([nvidia.com](https://developer.nvidia.com/cuda-downloads))

After installing TensorFlow, check that the GPU is found by running the following program:

```python
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```

The output should be something like:

```text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Usage

* Open the project
* If you are not using PyCharm, run `conda activate tf`
* Run `python process_data.py`
* Run `python pipeline.py`
