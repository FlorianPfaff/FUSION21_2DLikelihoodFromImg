# Based on code from https://github.com/activatedgeek/LeNet-5
# Modified for FUSION21 paper by Florian Pfaff, pfaff@kit.edu
from LeNet5.lenet import C1, C2, C3, F4
from torch import nn
from collections import OrderedDict

class F5NoLogSoftmax(nn.Module):
    def __init__(self, n_outputs):
        super(F5NoLogSoftmax, self).__init__()

        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, n_outputs))
        ]))

    def forward(self, img):
        a_b_unnorm = self.f5(img)
        return a_b_unnorm

class LeNet5NoLogSoftmax(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self, n_outputs):
        super(LeNet5NoLogSoftmax, self).__init__()

        self.c1 = C1()
        self.c2_1 = C2() 
        self.c2_2 = C2() 
        self.c3 = C3() 
        self.f4 = F4() 
        self.f5 = F5NoLogSoftmax(n_outputs=n_outputs) 

    def forward(self, img):
        output = self.c1(img)

        x = self.c2_1(output)
        output = self.c2_2(output)

        output += x

        output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        output = self.f5(output)
        return output