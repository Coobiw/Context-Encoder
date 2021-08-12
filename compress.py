

import argparse
import numpy as np
import torch

# from models.common import *

if __name__ == '__main__':

    weights_path = './model/uneven_with_edge_weighted.pth'
    is_half = True

    # Load pytorch model
    net = torch.load(weights_path, map_location=torch.device('cpu'))


    if is_half:
        net = net.half() # 把FP32转为FP16



    # Save .pt
    torch.save(net, './model/uneven_with_edge_weighted_compressed.pth')
