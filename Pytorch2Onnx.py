# import libraries
from torchvision import models
import torch

# create a dummy input with correct shape for the network
dummy_input = torch.randn(16, 3, 224, 224, device='cuda')

# created a resnet50 model
model = models.resnet50(pretrained=True).cuda()
model.eval()

# Created dynamic axes for dynamic batch_size not required for static batch_size
dynamic_axes = {"actual_input_1":{0:"batch_size"}, "output1":{0:"batch_size"}}
input_names = [ "actual_input_1" ]
output_names = [ "output1" ]

# Export the model to onnx
torch.onnx.export(model, dummy_input, "resnet50_dynamic.onnx", 
                  verbose=False,input_names=input_names,
                  output_names=output_names,dynamic_axes=dynamic_axes, export_params=True)
