import torch
w=torch.tensor([1.],requires_grad=True)
b=torch.tensor([2.],requires_grad=True)

a=torch.add(w,b)
a.retain_grad()#非叶子结点可以使用retain_grad()保存梯度
c=torch.add(w,1)
y=torch.mul(a,c)
y.backward()

print(a.grad)

print(w.grad)
print(b.grad)
 
print(w.is_leaf,b.is_leaf)#w和b是用户创建的叶子结点，非叶子结点的梯度在计算完成后会被释放掉
print(a.is_leaf,c.is_leaf,y.is_leaf)
#查看grad_fn
print("grad_fn:\n",w.grad_fn,b.grad_fn,a.grad_fn,c.grad_fn,y.grad_fn)
#动态图：运算与搭建同时进行 灵活易于调节
#静态图：先搭建图，后运算 高效单数不灵活