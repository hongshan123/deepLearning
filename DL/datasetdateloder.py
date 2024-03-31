#功能：构建可以迭代的数据装载器
#Epochs:所有训练样本都已经输入到模型中，称为一个epoch
#Iteration:一批样本输入模型中，称为一个Iteration
#Batchsize:批大小，决定一个Epoch有多少Iteration
# torch.utils.data.DateLoader(dataset,batch_size=1,
#                                    shuffle=False,
#                                    sampler=None,
#                                    batch_sampler=None,
#                                    num_workers=0
#                                    collate_fn=None,
#                                    pin_memory=False,
#                                    drop_last=False,
#                                    timeout=0,
#                                    work_init_fn=None,
#                                    multiprocessing_context=)
#dataset:Dateset类，决定数据从哪里读取以及如何读取
#batchsize:批大小
#num_works:是否进行多进程读取数据
#shuffl:是否打乱每个epoch
#drop_last:当样本数不能被batchsize整除时，是否舍弃最后一批数据

#torch.utils.data.Dataset
#功能:Dataset抽象类，所有自定义的Dataset需要继承它，并且复写__getitem__()
#getitem:接受一个索引，返回一个样本
# class Dataset(object):
    
#     def __getitem__(self,index):
#         raise NotImplementedError
    
#     def __add__(self,other):
#         return ConcatDataset([self,other])
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from PIL import Image
rmb_label = {"1": 0, "100": 1}
class rmbdataset(Dataset):
    def __init__(self,data_dir,transform=None):
        self.labelname={"1":0,"100":1}
        self.data_info=self.get_img_info(data_dir)
        self.transform=transform
    
    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        print(len(self.data_info))
        return len(self.data_info)
    @staticmethod
    def get_img_info(data_dir):
        data_info = list()#初始化一个空列表记录图片的路径与标签
         # 使用os.walk遍历data_dir目录及其所有子目录  
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            # 遍历当前目录下的所有子目录  
            for sub_dir in dirs:
                 # 获取子目录下的所有文件名  
                img_names = os.listdir(os.path.join(root, sub_dir))
                # 过滤出所有以.jpg结尾的图片文件名  
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info