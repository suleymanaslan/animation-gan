import numpy as np
import torch
import cv2
import imageio
from skimage.transform import resize
from datetime import datetime

import torchvision.datasets
import torchvision.transforms as transforms


def resize_batch(batch_img, size, batch_size):
    return resize(batch_img.permute(0, 2, 3, 1), (batch_size, size, size))


def get_img_cube(img_batch, img_i):
    cur_img = img_batch[img_i]
    cur_img_cube = None
    for j in range(4):
        cur_anim = cur_img[j*64:(j+1)*64]
        for i in range(9):
            cur_frame = cur_anim[:,i*64:(i+1)*64]
            cur_img_cube = cur_frame if cur_img_cube is None else np.concatenate((cur_img_cube, cur_frame), axis=2)
    return cur_img_cube	


def convert_to_img_plane(real_inputs, batch_size):
    real_img_cubes = None
    for batch_img_i in range(batch_size):
        cur_img = real_inputs[batch_img_i]
        cur_img_cube = None
        for anim_i in range(4):
            cur_anim = cur_img[:,anim_i*64:(anim_i+1)*64]
            for frame_i in range(9):
                cur_frame = cur_anim[:,:,frame_i*64:(frame_i+1)*64]
                cur_img_cube = cur_frame if cur_img_cube is None else torch.cat((cur_img_cube, cur_frame), dim=0)
        cur_img_cube = cur_img_cube.unsqueeze(0)
        real_img_cubes = cur_img_cube if real_img_cubes is None else torch.cat((real_img_cubes, cur_img_cube), dim=0)
    return real_img_cubes


def convert_to_img_cube(real_inputs, batch_size):
    real_img_cubes = None
    for batch_img_i in range(batch_size):
        cur_img = real_inputs[batch_img_i]
        cur_img_cube = None
        for anim_i in range(4):
            cur_anim = cur_img[:,anim_i*64:(anim_i+1)*64]
            for frame_i in range(9):
                cur_frame = cur_anim[:,:,frame_i*64:(frame_i+1)*64].unsqueeze(1)
                cur_img_cube = cur_frame if cur_img_cube is None else torch.cat((cur_img_cube, cur_frame), dim=1)
        cur_img_cube = cur_img_cube.unsqueeze(0)
        real_img_cubes = cur_img_cube if real_img_cubes is None else torch.cat((real_img_cubes, cur_img_cube), dim=0)
    return real_img_cubes


def preprocess_data(batch_size=64, data_size=40_000):
    assert data_size % batch_size == 0
    dataset = torchvision.datasets.ImageFolder(root="data", transform=transforms.Compose([transforms.ToTensor()]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    processed_data = torch.empty([data_size, 108, 64, 64], dtype=torch.uint8)
    for batch_index, (real_images, _) in enumerate(dataloader, 0):
        real_images = convert_to_img_plane((real_images * 255).type(torch.ByteTensor), batch_size)
        processed_data[batch_index*batch_size:(batch_index+1)*batch_size] = real_images
        if (batch_index + 1) % 10 == 0:
            print(f"{datetime.now()} {(batch_index+1)*batch_size}")
        if (batch_index+1)*batch_size >= data_size:
            break
    return processed_data


def preprocess_scaled_data(scale, batch_size=64, data_size=40_000):
    assert data_size % batch_size == 0
    
    img_sizes = {0: 4,
                 1: 8, 
                 2: 16, 
                 3: 32, 
                 4: 64}
    
    dataset = torchvision.datasets.ImageFolder(root="data", transform=transforms.Compose([transforms.ToTensor()]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    if scale >= 3:
        processed_data = torch.empty([data_size, 108, img_sizes[scale], img_sizes[scale]], dtype=torch.uint8)
    else:
        processed_data = torch.empty([data_size, 108, img_sizes[scale], img_sizes[scale]], dtype=torch.float32)
        normalize_t = torch.Tensor([0.5])
        
    for batch_index, (real_images, _) in enumerate(dataloader, 0):
        real_images = convert_to_img_plane((real_images * 255).type(torch.ByteTensor), batch_size)
        real_images = resize_batch(real_images, img_sizes[scale], batch_size)
        
        if scale >= 3:
            real_images = real_images * 255
            real_images = torch.from_numpy(real_images.transpose((0, 3, 1, 2))).type(torch.ByteTensor)
        else:
            real_images = (torch.from_numpy(real_images.transpose((0, 3, 1, 2))).type(torch.FloatTensor) - normalize_t) / normalize_t
        
        processed_data[batch_index*batch_size:(batch_index+1)*batch_size] = real_images
        
        if (batch_index + 1) % 10 == 0:
            print(f"{datetime.now()} {(batch_index+1)*batch_size}")
        if (batch_index+1)*batch_size >= data_size:
            break
    return processed_data


def animate_img_cube(input_img_cube, anim_file, training_outputs=False):
    frames = []
    for i in range(4*9):
        img_cube = input_img_cube[int((input_img_cube.shape[0]-1)*(i/35))][0] if training_outputs else input_img_cube
        append_img = ((img_cube[:,:,i*3:(i+1)*3]+1.0)*0.5*255).astype(np.uint8)
        append_img = cv2.resize(append_img, dsize=(0,0), fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
        frames.append(append_img)

    imageio.mimsave(anim_file, frames, 'GIF', fps=8)
    return anim_file


def animate_img_batch(img_batch, anim_file, get_img_cubes=True, training_outputs=False):
    if get_img_cubes:
        cube_0 = get_img_cube(img_batch, 0)
        cube_1 = get_img_cube(img_batch, 1)
        cube_2 = get_img_cube(img_batch, 2)
        cube_3 = get_img_cube(img_batch, 3)
    else:
        cube_0 = img_batch[0]
        cube_1 = img_batch[1]
        cube_2 = img_batch[2]
        cube_3 = img_batch[3]

    frames = []
    for i in range(4*9):    
        if training_outputs:
            cube_0 = img_batch[int((img_batch.shape[0]-1)*(i/35))][0]
            cube_1 = img_batch[int((img_batch.shape[0]-1)*(i/35))][1]
            cube_2 = img_batch[int((img_batch.shape[0]-1)*(i/35))][2]
            cube_3 = img_batch[int((img_batch.shape[0]-1)*(i/35))][3]
        
        append_img = ((cube_0[:,:,i*3:(i+1)*3]+1.0)*0.5*255).astype(np.uint8)
        append_img = cv2.resize(append_img, dsize=(0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        top_img = append_img
        append_img = ((cube_1[:,:,i*3:(i+1)*3]+1.0)*0.5*255).astype(np.uint8)
        append_img = cv2.resize(append_img, dsize=(0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        top_img = np.hstack((top_img, append_img))

        append_img = ((cube_2[:,:,i*3:(i+1)*3]+1.0)*0.5*255).astype(np.uint8)
        append_img = cv2.resize(append_img, dsize=(0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        bot_img = append_img
        append_img = ((cube_3[:,:,i*3:(i+1)*3]+1.0)*0.5*255).astype(np.uint8)
        append_img = cv2.resize(append_img, dsize=(0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        bot_img = np.hstack((bot_img, append_img))

        combined_img = np.vstack((top_img, bot_img))
        frames.append(combined_img)

    imageio.mimsave(anim_file, frames, 'GIF', fps=8)
    return anim_file


def get_sample_frames(anim_frames, from_gif=False, multiple=False, dim=64):
    if from_gif:
        if multiple:
            return np.hstack((np.vstack((anim_frames[0][:dim,:dim], anim_frames[0][:dim,dim:dim+dim], anim_frames[0][dim:dim+dim,:dim], anim_frames[0][dim:dim+dim,dim:dim+dim])), 
                              np.vstack((anim_frames[3][:dim,:dim], anim_frames[3][:dim,dim:dim+dim], anim_frames[3][dim:dim+dim,:dim], anim_frames[3][dim:dim+dim,dim:dim+dim])), 
                              np.vstack((anim_frames[9][:dim,:dim], anim_frames[9][:dim,dim:dim+dim], anim_frames[9][dim:dim+dim,:dim], anim_frames[9][dim:dim+dim,dim:dim+dim])), 
                              np.vstack((anim_frames[10][:dim,:dim], anim_frames[10][:dim,dim:dim+dim], anim_frames[10][dim:dim+dim,:dim], anim_frames[10][dim:dim+dim,dim:dim+dim])), 
                              np.vstack((anim_frames[18][:dim,:dim], anim_frames[18][:dim,dim:dim+dim], anim_frames[18][dim:dim+dim,:dim], anim_frames[18][dim:dim+dim,dim:dim+dim])), 
                              np.vstack((anim_frames[19][:dim,:dim], anim_frames[19][:dim,dim:dim+dim], anim_frames[19][dim:dim+dim,:dim], anim_frames[19][dim:dim+dim,dim:dim+dim])), 
                              np.vstack((anim_frames[27][:dim,:dim], anim_frames[27][:dim,dim:dim+dim], anim_frames[27][dim:dim+dim,:dim], anim_frames[27][dim:dim+dim,dim:dim+dim])), 
                              np.vstack((anim_frames[30][:dim,:dim], anim_frames[30][:dim,dim:dim+dim], anim_frames[30][dim:dim+dim,:dim], anim_frames[30][dim:dim+dim,dim:dim+dim]))))
        return np.hstack((anim_frames[0], anim_frames[3], anim_frames[9], anim_frames[10], anim_frames[18], anim_frames[19], anim_frames[27], anim_frames[30]))
    assert multiple == False
    anim_frames = anim_frames.numpy().transpose(1, 2, 0)
    return np.hstack((np.hstack((anim_frames[dim*0:dim*(0+1),dim*0:dim*(0+1)], anim_frames[dim*0:dim*(0+1),dim*3:dim*(3+1)])), 
                      np.hstack((anim_frames[dim*1:dim*(1+1),dim*0:dim*(0+1)], anim_frames[dim*1:dim*(1+1),dim*1:dim*(1+1)])), 
                      np.hstack((anim_frames[dim*2:dim*(2+1),dim*0:dim*(0+1)], anim_frames[dim*2:dim*(2+1),dim*1:dim*(1+1)])), 
                      np.hstack((anim_frames[dim*3:dim*(3+1),dim*0:dim*(0+1)], anim_frames[dim*3:dim*(3+1),dim*3:dim*(3+1)]))))

