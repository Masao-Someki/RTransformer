# RTransformer
Pytorch implementation of rnn-enhanced transformer.  
This is my practice to write in pytorch, so if anyone find my code is not easy to read, please teach me how I should write my code.

## Usage
You can use this rnn-enhanced transformer with the following codes.  
```
import torch
import json
from RT import get_rtrans

conf = json.load(open('rtrans.conf'))
net = get_rtrans(**conf)
net.eval()

data = torch.randn(1, 32, conf['d_model'])
# data should be formatted as (B, L, D)
# B as batch-size, L as sequence-length, D as feature-dimension.

out = net(data)
```
The shape of output is (B, L, D)
