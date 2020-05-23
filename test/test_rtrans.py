# This is a test script

import torch
import json
from RT import get_rtrans

def test_rtrans():
    conf = json.load(open('rtrans.conf'))
    net = get_rtrans(**conf)
    net.eval()

    data = torch.randn(1, 32, conf['d_model'])
    # data should be formatted as (B, L, D)
    # B as batch-size, L as sequence-length, D as feature-dimension.

    out = net(data)
    assert out.shape == (1, 32, conf['d_model'])
