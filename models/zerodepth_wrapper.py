import torch
from vidar.arch.models.zerodepth import ZeroDepth

class ZeroDepthWrapper:
    def __init__(self, checkpoint_path):
        self.model = ZeroDepth.load_from_checkpoint(checkpoint_path).eval().cuda()

    def predict(self, rgb):
        with torch.no_grad():
            return self.model(rgb)['depth']
