# Deeplab v3

This node is a modification of Tensorflow's Deeplab v3 for semantic segmentation inference code.

Tensorflow requires an specific environment to work, which might cause issues with ROS Python setup.
For this reason, we include the instructions for its use with a virtual environment.

This document uses `virtualenv`.

## Prerequisites

* Nvidia GPU
* Appropriate NVIDIA Driver version installed
* Corresponding CUDA+cuDNN version installed for the selected Tensorflow version. 

In this example: TF 1.12/CUDA9 

## Virtual environment setup

1. Create a container directory `tf_ros`. This directory will hold all the dependencies for Tensorflow.
`$ mkdir ~/tf_ros && cd ~/tf_ros`
1. Create the virtual environment for Python2 named `tensor_ros_deeplab`.
`$ virtualenv --system-site-packages -p python2 tensor_ros_deeplab`
1. Activate the environment.
`$ source tensor_ros_deeplab/bin/activate`
1. Download Tensorflow for Python 2.7 with GPU support from the Tensorflow downloads page.
https://www.tensorflow.org/install
1. Setup Tensorflow in the environment. 
`(tensor_ros_deeplab) $ pip install --upgrade tensorflow_gpu-1.12.0-cp27-none-linux_x86_64.whl`
1. Test that Tensorflow is working in the venv.
`(tensor_ros_deeplab) $ python -c "import tensorflow as tf; print(tf.__version__)"`
1. The terminal should display the Tensorflow version.
1. Go to the Autoware installation directory and source the workspace.

## How to launch

In a sourced terminal, inside the virtual environment:
1. `(tensor_ros_deeplab) $ roscd vision_deeplab_segment`
1. `(tensor_ros_deeplab) vision_deeplab_segment$ python scripts/vision_deeplab_segment.py PATH_TO_MODEL`

Replace `PATH_TO_MODEL` to the actual path of the pretrained tensorflow tar file.

## Parameters

|Parameter| Type| Description|
----------|-----|--------
|`model_path`|Positional, *String* |Path to the tar file containing the frozen Tensorflow pretrained model.|
|`--image_src`|Flag, *String*|Non-Maximum suppresion area threshold ratio to merge proposals. Default `/image_raw`.|
|`--label_format`|Flag, *String*|Label format of the provided model. Currently `pascal` or `cityscapes`. Default `pascal`.|
