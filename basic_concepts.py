import torch
import numpy as np
#from __future__ import print_function

x = torch.empty(5,3)
# y = torch.rand(5,3)
# z = torch.zeros(5,3, dtype=torch.long)
# w = torch.tensor([[1,2,3],[5,6,7],[7,8,9]]) #create tensor by data
# ww = w.new_ones(1,2,3,dtype = torch.double)
# xx = torch.randn_like(x, dtype=torch.float) #same size as x , but different dType
# #print(xx.size()) #torch.Size([5, 3])is a tuple
# sum1 = x + y # add method 1
# sum2 = torch.empty(5,3)
# torch.add(x,y,out = sum2)# add method 2
# y.add_(x) # add method 3    function_ "_"表示原地操作（inplace) change y directely
# # print(w)
# #-----索引操作-----
# # print(w[:1]) # print(w[0,:]) row 0 ->row 1 but not include row 1  [1,2,3]
# # print(w[:,:1]) # get all rows and 1st column

# #-----改变形状view()-----
# x_size1 = x.view(15) #change shape to a line
# x_size2 = x.view(-1,5)  # should be(3,5), -1 means infer from other dimensions 
# print("x is ：{}\nsize1 is :{} \nsize2 is :{} \n".format(x,x_size1,x_size2))

# # torch tensor <-->numpy ndarray 张量和numpy数组的转换
# # tensor to numpy
# a = torch.ones(5)
# a.add_(2) #add 2 to each element of a
# print(a) #tensor [...]
# b = a.numpy()
# print(b) # [...]

# # numpy to tensor
# a = np.ones(5) #[1...]
# b = torch.from_numpy(a) #tensor [1...]
# b.add_(2) # tensor[3...]
# print(b)

#---CUDA上的张量 tensor on cuda-----
if torch.cuda.is_available():
    device = torch.device("cuda")  #cuda device object 
    y = torch.ones_like(x, device=device) #create a tensor on GPU 在GPU创建一个tensor
    x = x.to(device)  # == x.to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu",torch.float64))  #从GPU改回到CPU
    
