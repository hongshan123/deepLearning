import torch
#torch.autograd.backward(tensors,grad_tensors=None,retain_graph=None,create_graph=False)
#tensors:用于求导的张量，如loss，retain_graph：保存计算图 ，create_graph：创建导数计算图，用于高阶求导 
#grad_tensors：多梯度权重
flag=True
flag=False
if flag:
    w=torch.tensor([1.],requires_grad=True)
    x=torch.tensor([2.],requires_grad=True)
    
    a=torch.add(w,x)
    b=torch.add(w,1)
    y=torch.mul(a,b)
    
    y.backward(retain_graph=True)
    y.backward()
    print(w.grad)
flag=True
flag=False
if flag:
    w=torch.tensor([1.],requires_grad=True)
    x=torch.tensor([2.],requires_grad=True)
    
    a=torch.add(w,x)
    b=torch.add(w,1)
    
    y0=torch.mul(a,b)#y0=(x+w)*(w+1)
    y1=torch.add(a,b)#y1=(x+w)+(w+1)
    
    loss=torch.cat([y0,y1],dim=0)
    
    grad_tensor=torch.tensor([1.,2.])
    
    loss.backward(gradient=grad_tensor)#gradien传入torch.autograd.backward()中的grad_tensors
    print(w.grad)
flag=True
flag=False
if flag:
    
    x=torch.tensor([3.],requires_grad=True)
    y=torch.pow(x,2)
    
    grad_1=torch.autograd.grad(y,x,create_graph=True)#只有创建导数的计算图才能求高阶导
    #grad_1=dy/dx=2x=6
    print(grad_1)
    
    grad_2=torch.autograd.grad(grad_1,x)
    #grad_2=d(2x)/dx=2
    print(grad_2)
flag=True
flag=False
if flag:
    w=torch.tensor([1.],requires_grad=True)
    x=torch.tensor([2.],requires_grad=True)
    
    for i in range(10):
        a=torch.add(w,x)
        b=torch.add(w,1)
        y=torch.mul(a,b)
        
        y.backward()
        print(w.grad)
        w.grad.zero_()#pytorch中梯度不会自动清零，需要手动清除，不然会在一次次求导中叠加
flag=True
flag=False  
#依赖于叶子结点的结点，requires_grads默认为true     
if flag:
    w=torch.tensor([1.],requires_grad=True)
    x=torch.tensor([2.],requires_grad=True)
    
    a=torch.add(w,x)
    b=torch.add(w,1)
    y=torch.mul(a,b)
    
    print(a.requires_grad,b.requires_grad,y.requires_grad)
#叶子结点不可执行in-place操作,即在原始内存中改变数据
flag=True
# flag=False      
if flag:
    w=torch.tensor([1.],requires_grad=True)
    x=torch.tensor([2.],requires_grad=True)
    
    a=torch.add(w,x)
    b=torch.add(w,1)
    y=torch.mul(a,b)
    
    # w.add_(1)
    #无法实现a leaf Variable that requires grad is being used in an in-place operation.
    
    y.backward()

