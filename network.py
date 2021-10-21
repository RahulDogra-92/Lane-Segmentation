import torch
import os
import torch.nn as nn
import torchvision

class SegnetV2(nn.Module):
    def __init__(self, chkpt = ",models"): # checkpoint to save the model
        super(SegnetV2, self).__init__()
        self.file = os.path.join(chkpt, "segnet_v2")
        self.base_model = torchvision.models.segmentation.deeplabv3_resnet101(False, num_classes=5)

    def forward(self, x):
        return self.base_model(x)['out'] # we just want regular output

    def save(self): #save the model
        torch.save(self.state_dict(), self.file)

    def load(self): # load the model
        self.load_state_dict(torch.load(self.file))

