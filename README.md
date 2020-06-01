# resnest tensorRT inference using torch2trt

## modify code

```
class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = torch.transpose(x.view(batch, self.cardinality, self.radix, -1),1,2)   #.permute(0,2, 1, 3) ***
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x
```

## requirement

1. torch2trt
2. torch

## speed test

```
python test.py 

show first 10 output:
pytorch: tensor([-0.8604,  0.3855,  0.3289,  0.1835,  0.7670, -0.2655, -0.4620, -0.5909,
        -0.9840,  0.3266], device='cuda:0')
tensorRT: tensor([-0.8604,  0.3855,  0.3289,  0.1835,  0.7670, -0.2655, -0.4620, -0.5909,
        -0.9840,  0.3266], device='cuda:0')


error: tensor(0.0024, device='cuda:0')


3x224x224 image inference time:
pytorch: 0.016407443969510497
tensorRT: 0.009609003949444741

```


