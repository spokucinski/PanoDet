import os
import subprocess
import timeit
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.chdir("Model_Exporter")
model_name = "saved_model"
proc = subprocess.run('python -m tf2onnx.convert --saved-model "C:\\Users\\Sebastian\\Documents\\PanoDet\\Model_Exporter\\input\\vgg19" --output "C:\\Users\\Sebastian\\Documents\\PanoDet\\Model_Exporter\\output\\exported_model.onnx" --opset 11 --inputs-as-nchw input_1'.split(), capture_output=True)
print(proc.returncode)
print(proc.stdout.decode('ascii'))
print(proc.stderr.decode('ascii'))