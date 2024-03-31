import torch
flag=True
#torch.cat()拼接,将张量按维度dim进行拼接
# if flag:
#     t=torch.ones((4,3))
#     t_0=torch.cat([t,t],dim=0)
#     t_1=torch.cat([t,t],dim=1)
#     print("t_0:{} shape:{}\nt_1:{} shape:{}".format(t_0,t_0.shape,t_1,t_1.shape))
#torch.stack()拼接会添加一个纬度
# if flag:
#     t=torch.ones((4,3))
#     t_0=torch.stack([t,t],dim=0)
#     t_1=torch.stack([t,t],dim=1)
#     print("t_0:{} shape:{}\nt_1:{} shape:{}\n".format(t_0,t_0.shape,t_1,t_1.shape))
#     t_stack=torch.stack([t,t],dim=2)
#     print("t_stack:{} shape:{}".format(t_stack,t_stack.shape))
#torch.chunk()，将张量按维度dim进行平均切分，返回张量列表（imput，chunk，dim），切分的张量，切分的份数，切分的维度
# if flag:
#     a=torch.ones((2,7))
#     list_of_tensor=torch.chunk(a,dim=1,chunks=2)
#     for idx,t in enumerate(list_of_tensor):
#         print("第{}个张量：{},shape is {}".format(idx+1,t,t.shape))
#torch.split(tensor,split_size_or_section,dim)为int时表示每一份的长度，为list时，按list元素切分
# if flag:
#     a=torch.ones((2,5))
#     # list_of_tensor=torch.split(a,2,dim=1)
#     list_of_tensor=torch.split(a,[1,4],dim=1)
#     for idx,t in enumerate(list_of_tensor):
#         print("第{}个张量：{},shape is {}".format(idx+1,t,t.shape))
#torch.index()索引元素
# if flag:
#     t=torch.randint(0,9,size=(3,3))
#     idx=torch.tensor([0,2],dtype=torch.long)
#     t_select=torch.index_select(t,dim=0,index=idx)
#     print("t:\n{}\nt_select:\n{}".format(t,t_select))
#torch.masked_select(input,mask,out=None)按照mask中的true进行索引，返回值为一维张量
#input为要索引的张量，mask与input同形状的布尔型张量
# if flag:
#     t=torch.randint(0,9,size=(3,3))
#     mask=t.le(5)#还有ge，gt，le，lt
#     t_select=torch.masked_select(t,mask)
#     print("t:\n{}\nt_select:\n{}mask:{}\n".format(t,t_select,mask))
#张量变换
#torch.reshape(input,shape) input为要变换的张量，shape是新张量的形状//当张量在内存中是连续的，新张量与input共享数据内存
# if flag:
#     t=torch.randperm(8)
#     t_reshape=torch.reshape(t,(-1,2,2))
#     print("t:{}\n,t_reshape:{}\n".format(t,t_reshape))
#     t[0]=1024
#     # print(t.data)
#     print("t:{}\n,t_reshape:{}\n".format(t,t_reshape))
#     print("t.data内存地址为:{}".format(id(t.data)))#id是索引地址类似于c中的&
#     print("t_reshape.data内存地址为:{}".format(id(t_reshape.data)))

#torch.transpose(input,dim0,dim1) input:变换的张量 dim0/dim1:要交换的维度
#矩阵转置等价于torch.transpose(input,0,1)==torch.t(input)
# if flag:
#     t=torch.rand((2,3,4))
#     t_transpose=torch.transpose(t,dim0=1,dim1=2)
#     print("t:{}\nt_transpose:{}".format(t,t_transpose))
#     print("t:{}\nt_transpose:{}".format(t.shape,t_transpose.shape))

#torch.squeeze(imput,dim=None,out=None) 压缩长度为1的维度 dim:若为None，移除所有长度为1的轴，若指定维度，只有指定维度长度为1时可以被移除；
#torch.unsqueeze(input,dim,out=None) 依据dim扩展维度
# if flag:
#     t=torch.rand((1,2,3,1))
#     t_sq=torch.squeeze(t)
#     t_0=torch.squeeze(t,dim=0)
#     t_1=torch.squeeze(t,dim=1)
#     print(t.shape)
#     print(t_sq.shape)
#     print(t_0.shape)
if flag:
    t=torch.rand((1,2,3,4))
    t_unsqueeze=torch.unsqueeze(t,dim=0)
    print(t_unsqueeze.shape)