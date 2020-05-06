import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx_coreml import convert

import torch
import torch.onnx
from onnx_coreml import convert
# from pytorch_transformers import *
import numpy as np

# # Step 0 - (a) Define ML Model
# class small_model(nn.Module):
#     def __init__(self):
#         super(small_model, self).__init__()
#         self.fc1 = nn.Linear(768, 256)
#         self.fc2 = nn.Linear(256, 10)

#     def forward(self, x):
#         y = F.relu(self.fc1(x))
#         y = F.softmax(self.fc2(y))
#         return y

# Model path
TMP_DIR = './'
model_name = 'pose_resnet_101_256x192'
pt_path = TMP_DIR + model_name + '.pth'
onnx_model_path = TMP_DIR + model_name + '.onnx'
mlmodel_path  = TMP_DIR + model_name + '.mlmodel'

import ipdb; ipdb.set_trace()
model = torch.load(pt_path, map_location=torch.device('cpu'))


# Step 1 - Convert from PyTorch to ONNX
test_input = torch.randint(0, 512, (1, 512))
torch.onnx.export(model,
                  test_input,
                  onnx_model_path,
                  input_names=["input_ids"],
                  output_names=["start_scores", "end_scores"])


exit()
# Step 0 - (b) Create model or Load from dist
in_model_path = './pose_resnet_101_256x192.pth'
model = torch.load(in_model_path)
dummy_input = torch.randn(768)

# Step 1 - PyTorch to ONNX model
torch.onnx.export(model, dummy_input, './small_model.onnx')

# Step 2 - ONNX to CoreML model
mlmodel = convert(model='./small_model.onnx', minimum_ios_deployment_target='13')
# Save converted CoreML model
mlmodel.save('small_model.mlmodel')