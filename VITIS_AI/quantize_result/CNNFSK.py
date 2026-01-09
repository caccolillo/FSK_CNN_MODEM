# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class CNNFSK(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(CNNFSK, self).__init__()
        self.module_0 = py_nndct.nn.Input() #CNNFSK::input_0
        self.module_1 = py_nndct.nn.Conv1d(in_channels=2, out_channels=1, kernel_size=[1], stride=[1], padding=[0], dilation=[1], groups=1, bias=True) #CNNFSK::CNNFSK/Conv1d[conv]/92
        self.module_2 = py_nndct.nn.MaxPool1d(kernel_size=[2], stride=[2], padding=[0], dilation=[1], ceil_mode=False) #CNNFSK::CNNFSK/MaxPool1d[maxpool]/102
        self.module_3 = py_nndct.nn.Module('nndct_shape') #CNNFSK::CNNFSK/104
        self.module_4 = py_nndct.nn.Module('nndct_reshape') #CNNFSK::CNNFSK/109
        self.module_5 = py_nndct.nn.Linear(in_features=4, out_features=2, bias=True) #CNNFSK::CNNFSK/Linear[fc]/110

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_3 = self.module_3(input=output_module_0, dim=0)
        output_module_4 = self.module_4(input=output_module_0, shape=[output_module_3,-1])
        output_module_4 = self.module_5(output_module_4)
        return output_module_4
