import torch
x=torch.rand((100,1))*10
y=x*2+(5+torch.randn(100,1))
#y=x*2+5
#创建训练数据
#构建线性回归参数
w=torch.randn((1),requires_grad=True)
b=torch.randn((1),requires_grad=True)
lr=0.0001
for iteration in range(1000000):
    #前向传播
    wx=torch.mul(w,x)
    y_pred=torch.add(wx,b)
    #计算mse loss
    loss=(0.5*(y-y_pred)**2).mean()
    #反向传播
    loss.backward()
    #更新参数
    b.data.sub_(lr*b.grad)
    w.data.sub_(lr*w.grad)
    b.grad.zero_()
    w.grad.zero_()
    if(loss<1):
        print("这是第{}轮训练，loss为{},w为{},b为{}".format(iteration,loss,w,b))