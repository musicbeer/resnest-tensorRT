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

## speed test(gtx1050ti)

```
python test.py 

----------------------resnest 50 test---------------------
show first 10 output:
pytorch: tensor([-0.8604,  0.3855,  0.3289,  0.1835,  0.7670, -0.2655, -0.4620, -0.5909,
        -0.9840,  0.3266], device='cuda:0')
tensorRT: tensor([-0.8604,  0.3855,  0.3289,  0.1835,  0.7670, -0.2655, -0.4620, -0.5909,
        -0.9840,  0.3266], device='cuda:0')


error: tensor(0.0024, device='cuda:0')


3x224x224 image inference time:
pytorch: 0.016652769159991296
tensorRT: 0.009595841290429235
----------------------resnet 50 test---------------------
show first 10 output:
pytorch: tensor([-0.3081,  0.0798, -1.1900, -1.4837, -0.5136,  0.3683, -2.1639, -0.8705,
        -1.8812, -0.1608], device='cuda:0')
tensorRT: tensor([-0.3081,  0.0798, -1.1900, -1.4837, -0.5136,  0.3683, -2.1639, -0.8705,
        -1.8812, -0.1608], device='cuda:0')


error: tensor(0.0011, device='cuda:0')


3x224x224 image inference time:
pytorch: 0.01067693106131628
tensorRT: 0.005997186410240829
----------------------resnet 101 test---------------------
show first 10 output:
pytorch: tensor([-0.9190, -0.1303, -1.0331, -1.5215, -0.5173,  0.1020, -2.0780, -0.8331,
        -1.4745, -0.4834], device='cuda:0')
tensorRT: tensor([-0.9190, -0.1303, -1.0331, -1.5215, -0.5173,  0.1020, -2.0780, -0.8331,
        -1.4745, -0.4834], device='cuda:0')


error: tensor(0.0013, device='cuda:0')


3x224x224 image inference time:
pytorch: 0.019209207659587266
tensorRT: 0.011350022009573878




```
resnest50 trt speedup 16ms->9.6ms, resnet50 11ms->6.1ms,resnet101 19ms->11ms,so resnest50 is faster than resnet101.


