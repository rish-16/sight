<p align="center">
    <br>
	<img src="./Assets/logo.png" width=200>
    <br>
<p>

<p align="center">
    <a href="https://github.com/rish-16/sight/blob/master/LICENSE">
		<img alt="AUR license" src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg">
    </a>
</p>

<h3 align="center">
<p>State-of-the-art Computer Vision and Object Detection for TensorFlow 1.15.</p>
</h3>

*Sight* provides state-of-the-art general-purpose architectures (YOLO9000, MaskRCNN, Fast/Faster RCNN, SSD...) for Computer Vision and Object Detection tasks with 30+ pretrained models written in TensorFlow 1.15.

## Installation

`sight` is written in Python 3.5+ and TensorFlow 1.15. 

Ideally, `sight` should be installed in a [virtual environments](https://docs.python.org/3/library/venv.html). If you're unfamiliar with Python virtual environments, check out this [tutorial](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) on getting started.

### Via PyPi

To use `sight`, you must first have TensorFlow installed. To do so, follow the instructions on the [TensorFlow installation page](https://www.tensorflow.org/install/pip?lang=python3).

When your virtual environment is set up with TensorFlow, you can install `sight` using `pip`:

```bash
pip install sight
```

### From Source

Again, to install from source, you need TensorFlow 1.15 and above running in a virtual environment. You can install the package by cloning the repo and installing the dependencies:

```bash
git clone https://github.com/rish-16/sight
cd sight
pip install .
```

### Model Architectures

1. Mask RCNN
2. YOLOv3 (Darknet by Joseph Redmon)