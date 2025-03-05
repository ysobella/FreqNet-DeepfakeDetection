import os
import time
import torch
import numpy as np
from util import Logger
from validate import validate
from networks.freqnet import freqnet
from options.test_options import TestOptions

opt = TestOptions().parse(print_options=False)
print(f'Model Path: {opt.model_path}')

# Load model
model = freqnet(num_classes=1)
state_dict = torch.load(opt.model_path, map_location='cpu')

# Ensure correct state_dict loading
if 'model' in state_dict:
    model.load_state_dict(state_dict['model'])
else:
    model.load_state_dict(state_dict)

model.cuda()
model.eval()

# Set test dataset path
test_dataroot = os.path.join(opt.dataroot, 'test')

# Create test options
test_opt = TestOptions().parse(print_options=False)
test_opt.dataroot = test_dataroot
test_opt.classes = ['real', 'fake']

# Run validation on test set
print("Evaluating on Test Set...")
acc, ap, r_acc, f_acc, _, _ = validate(model, test_opt)

print(f"Test Accuracy: {acc:.4f}")
print(f"Test AP: {ap:.4f}")
print(f"Accuracy of Real Images: {r_acc:.4f}" if r_acc is not None else "No real images in test set")
print(f"Accuracy of Fake Images: {f_acc:.4f}" if f_acc is not None else "No fake images in test set")
