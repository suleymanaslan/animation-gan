import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import os


def calculate_anim_mask(batch_size):
    if os.path.exists(f"utils/anim_mask_{batch_size}.npy"):
        anim_mask = np.load(f"utils/anim_mask_{batch_size}.npy")
        if anim_mask.shape[0] == batch_size:
            return torch.from_numpy(anim_mask)
        
    print(f"Creating anim_mask, batch_size:{batch_size}")
    
    dataset = torchvision.datasets.ImageFolder(root="data", transform=transforms.Compose([transforms.ToTensor(),
                                                                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    anim_mask = np.ones((64, 64))
    for batch_index, (real_inputs, _) in enumerate(dataloader, 0):
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
        anim_mask = np.logical_and(anim_mask, (real_img_cubes.mean(axis=0).mean(axis=0).numpy() == 1)).astype(np.uint8)
    anim_mask[:1,:] = 1
    
    return torch.from_numpy(np.tile(anim_mask, (batch_size, 108, 1, 1))) == 1

