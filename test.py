from test.resnest import resnest50
import torch
from torch2trt import torch2trt 
from timeit import default_timer as timer
net = resnest50(pretrained=True).cuda().eval()
x = torch.ones((1, 3, 224, 224))

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(net, [x.cuda()])
with torch.no_grad():
    y_pt = net(x.cuda())[0]
y_trt = model_trt(x.cuda())[0]
print('show first 10 output:')
print('pytorch:',y_pt[:10])
print('tensorRT:',y_trt[:10])
print('\n')
print('error:',sum(abs(y_trt-y_pt)))
print('\n')
x=x.cuda()
print('3x224x224 image inference time:')
start= timer()
for i in range(100):
    net(x.cuda())
end = timer()-start
print('pytorch:',end/100.)
start= timer()
for i in range(100):
    model_trt(x)
end = timer()-start
print('tensorRT:',end/100.)


