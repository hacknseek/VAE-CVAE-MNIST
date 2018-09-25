import torch
from torch.autograd import Variable
import os
import numpy as np
import scipy
import scipy.misc

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        # print('cuda')
        x = x.cuda()
    return Variable(x, volatile=volatile)

def idx2onehot(idx, n):

    assert idx.size(1) == 1
    assert torch.max(idx).data[0] < n

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx.cpu().data, 1)
    onehot = to_var(onehot)
    
    return onehot

def save_img(args, x, num_iter, recon=False):
    display_row = 5
    display_col = 5

    if not(os.path.exists(args.figroot)):
        os.mkdir(os.path.join(args.figroot))
    save_root = os.path.join(args.figroot, args.data)
    if not(os.path.exists(save_root)):
        os.mkdir(os.path.join(save_root))
    path_recon = os.path.join(save_root, 'recon')
    if not(os.path.exists(path_recon)):
        os.mkdir(path_recon)

    img_sz = args.img_size
    if args.img_channel == 1:
        fig_img = np.zeros((img_sz*display_row, img_sz*display_col))
    else:
        fig_img = np.zeros((img_sz*display_row, img_sz*display_col, 3))
    
    x = x.view(x.size(0), args.img_channel, img_sz, img_sz).data.cpu().detach().numpy()
    x = x.transpose(0, 2, 3, 1)

    for row in range(display_row):
        for col in range(display_col):
            t = row*args.num_labels + col
            if t >= x.shape[0]:
                continue

            if args.img_channel == 1:
                fig_img[row*img_sz:(row+1)*img_sz, col*img_sz:(col+1)*img_sz] = x[t,:,:,0]
            else:
                fig_img[row*img_sz:(row+1)*img_sz, col*img_sz:(col+1)*img_sz, :] = x[t,:,:,:]
            if recon == False:
                scipy.misc.imsave(os.path.join(save_root, str(num_iter)+'.jpg'), fig_img)
            else:
                scipy.misc.imsave(os.path.join(path_recon, str(num_iter)+'_r.jpg'), fig_img)

