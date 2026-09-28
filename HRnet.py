import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
High-Resolution NN 
We have a few branches in different resolutions - 32x32, 64x64 and so on
and the idea is - we use the lowe-reoslution branches to get the context
and the high-resolution branch will get us the finer details 
The flow is -> the few branches forward the image in different resolution parallely
and after each layer they pass every branch block with every another branch block 
FINAL = D1 + Downsample(D2) + Upsample(D3) -> so to get the outto be the same resolution we
upscale the lower-res layer and upscale the high-res layer
'''

# convBlock - this is our building block
class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.rel1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 3, kernel_size=3, padding=1)
        self.rel2 = nn.ReLU()

    def forward(self, x):
        return self.rel2(self.conv2(self.rel1(self.conv1(x))))

# This is our branch - made of convBlocks
class Branch(nn.Module):
    def __init__(self, num_blocks, size):
        super().__init__()
        self.blocks = [ConvBlock() for num in range(num_blocks) ]
        self.size = size # this is the size of the resolution
        self.num_blocks = num_blocks

    def forward(self, x):
        out = self.blocks[0](x)
        for i in range(1, len(self.blocks)):
            out = self.blocks[i].forward(out)
        return out
# HRnet
class HighResolutionNet(nn.Module):
    def __init__(self, blocks_count : list, sizes : list):
        super().__init__()
        self.blocks_count = blocks_count
        self.branches = [Branch(num, size) for num, size in zip(blocks_count, sizes)]

    def forward(self, x):
        inputs = [F.interpolate(x, scale_factor=1/2**i, mode='bilinear') for i in range(len(self.branches))]
        # first we interpolate the input image -> so to match the resolution for each branch
        maxi = max(self.blocks_count)
        diff = maxi
        sizes = list(self.branches[i].size for i in range(len(self.branches)))
        
        for i in range(maxi):
            nums = []
            ''' 
            The idea here is we get the max_blocks count and in each iteration we substract the diff from the branch's blocks_count
            and we decrement the diff with 1 after each iter, so if blocksCount - diff >= 0 means we can call the forward method - it will be valid
            and then we add this blocks idx to the nums list - and then if is only one element we perceed otherwise we add the outs from the other branches
            and we combine the results from the other branches
            '''
            for j in range(len(self.branches)):
                if self.blocks_count[j] - diff >= 0:
                    # in the inputs list we keep all the results/outs from the previous layers and we overwrite on each iteration

                    out = self.branches[j].forward(inputs[j])
                    inputs[j] = out
                    nums.append(j)
                
                if len(nums) <= 1:
                    continue

                for idx1 in range(0, len(nums) - 1):
                    for idx2 in range(1, len(nums)):
                        c = self.branches[idx1].size / self.branches[idx2].size 
                        # we get the scale_factor if it is 2 times bigger idx1 from the idx2 - .size gets us the resolution
                        inputs[idx1] += F.interpolate(inputs[idx2], scale_factor=c, mode='bilinear') # here we interpolate with the scale factor
                        inputs[idx2] += F.interpolate(inputs[idx1], scale_factor=1/c, mode='bilinear')

            diff -= 1 # we decrement the counter
        #finally we return the final overwritten inputs - these will be the final calculated results and sums from the other branches
        
        return inputs

hr = HighResolutionNet([3, 2, 1], [64, 32, 16])
x = torch.randn([1, 3, 64, 64])
print(hr.forward(x)[2].shape)

   # def forward(self, x):
    #     input = x
    #     max_num = max(self.blocks_count)

    #     dir_branches = {}
    #     dir_out = {}

    #     for i in range(self.blocks_count):
    #         if i == 0:
    #             dir[i] = x
    #         dir[i] = F.interpolate(x, scale_factor=0.5, mode='bilinear')

    #     for i, b in enumerate(self.blocks_count):
    #         dir_branches[i] = max_num - b

    #     for i in range(len(max_num)):
    #         for j in range(len(self.branches)):
    #             idx = i - dir_branches[j]
            
    #         for j in range(len(self.branches)):
    #             idx = i - dir_branches[j]

    #             if idx < 0:
    #                 continue
    #             elif idx == 0:
    #                 out = x
    #                 self.branches[i].blocks[j].forward_by_idx(out)
    #             else:
    #                 for l, k in enumerate(self.branches):
    #                     if l == j:
    #                         continue
    #                     if k[i].size == k.size:
    #                          += k.forward()  

    #                     out += F.interpolate
    #             dir[i] = out

    #     for n in self.branches:
    #         out = n.forward(input)
    #         input = out
    #     return out

    # def forward_by_idx(self, x, idx):
    #             out = self.blocks[idx](x)
    #             return out
    
