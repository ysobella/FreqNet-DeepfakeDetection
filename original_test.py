import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.freqnet import freqnet
from options.test_options import TestOptions
import numpy as np
import random


DetectionTests = {
                'ForenSynths': { 'dataroot'   : '/opt/data/private/DeepfakeDetection/ForenSynths/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },

           'GANGen-Detection': { 'dataroot'   : '/opt/data/private/DeepfakeDetection/GANGen-Detection/',
                                 'no_resize'  : True,
                                 'no_crop'    : True,
                               },

                 }


opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')

# get model
model = freqnet(num_classes=1)

# from collections import OrderedDict
# from copy import deepcopy
# state_dict = torch.load(opt.model_path, map_location='cpu')['model']
# pretrained_dict = OrderedDict()
# for ki in state_dict.keys():
    # pretrained_dict[ki[7:]] = deepcopy(state_dict[ki])
# model.load_state_dict(pretrained_dict, strict=True)

model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
model.cuda()
model.eval()


for testSet in DetectionTests.keys():
    dataroot = DetectionTests[testSet]['dataroot']
    printSet(testSet)

    accs = [];aps = []
    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    for v_id, val in enumerate(os.listdir(dataroot)):
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.classes  = '' #os.listdir(opt.dataroot) if multiclass[v_id] else ['']
        opt.no_resize = DetectionTests[testSet]['no_resize']
        opt.no_crop   = DetectionTests[testSet]['no_crop']
        acc, ap, _, _, _, _ = validate(model, opt)
        accs.append(acc);aps.append(ap)
        print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
    print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25)
