import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

import utils


def print_and_log(text, model_dir):
    print(text)
    print(text, file=open(f'{model_dir}/log.txt', 'a'))


def train_dataloader(discriminator_net, generator_net, discriminator_optimizer, generator_optimizer, model_dir, dataloader, anim_mask, latent_size, num_epochs, batch_size):

    device = torch.device("cuda:0")
    
    criterion = nn.BCELoss()
    
    fixed_noise2d = torch.randn(4, latent_size, 1, 1, device=device)
    # fixed_noise3d = torch.randn(4, latent_size, 1, 1, 1, device=device)
    
    real_label = torch.full((batch_size,), 1, device=device)
    fake_label = torch.full((batch_size,), 0, device=device)
    
    generator_losses = []
    discriminator_losses = []
    generated_img_cubes = None
    train_iters = 0
    torch_ones = torch.tensor([1.0]).to(device)
    
    for epoch in range(num_epochs):
        for batch_index, (real_images, _) in enumerate(dataloader, 0):

            discriminator_net.zero_grad()

            # 3d data
            # real_inputs = utils.convert_to_img_cube(real_images, batch_size).to(device)
            # 2d data
            real_inputs = utils.convert_to_img_plane(real_images, batch_size).to(device)
            # spatio-temporal data
            # real_inputs = utils.convert_to_img_plane(real_images, batch_size).reshape(-1, 3, 64, 64).to(device)

            output = discriminator_net(real_inputs).view(-1)
            discriminator_loss_real = criterion(output, real_label)

            discriminator_loss_real.backward()
            real_output = output.mean().item()

            noise2d = torch.randn(batch_size, latent_size, 1, 1, device=device)
            # noise3d = torch.randn(batch_size, latent_size, 1, 1, 1, device=device)

            generated_inputs = generator_net(noise2d)
            # generated_inputs = generator_net(noise3d)
            generated_inputs = torch.where(anim_mask, torch_ones, generated_inputs)

            # discriminator:3D
            # output = discriminator_net(generated_inputs.detach().reshape(-1, 3, 36, 64, 64)).view(-1)
            # discriminator:2D
            output = discriminator_net(generated_inputs.detach().reshape(-1, 108, 64, 64)).view(-1)
            # discriminator:spatio-temporal
            # output = discriminator_net(generated_inputs.detach().reshape(-1, 3, 64, 64)).view(-1)

            discriminator_loss_fake = criterion(output, fake_label)

            discriminator_loss_fake.backward()
            fake_output1 = output.mean().item()

            discriminator_loss = discriminator_loss_real + discriminator_loss_fake

            discriminator_optimizer.step()

            generator_net.zero_grad()

            # discriminator:3D
            # output = discriminator_net(generated_inputs.reshape(-1, 3, 36, 64, 64)).view(-1)
            # discriminator:2D
            output = discriminator_net(generated_inputs.reshape(-1, 108, 64, 64)).view(-1)
            # discriminator:spatio-temporal
            # output = discriminator_net(generated_inputs.reshape(-1, 3, 64, 64)).view(-1)

            generator_loss = criterion(output, real_label)

            generator_loss.backward()
            fake_output2 = output.mean().item()

            generator_optimizer.step()

            if batch_index % 250 == 0:
                print_and_log(f"{datetime.now()} [{epoch:02d}/{num_epochs}][{batch_index:04d}/{len(dataloader)}]\t"
                              f"D_Loss:{discriminator_loss.item():.4f} G_Loss:{generator_loss.item():.4f} Real:{real_output:.4f} "
                              f"Fake1:{fake_output1:.4f} Fake2:{fake_output2:.4f}", model_dir)

            generator_losses.append(generator_loss.item())
            discriminator_losses.append(discriminator_loss.item())

            if (train_iters % 2500 == 0) or ((epoch == num_epochs-1) and (batch_index == len(dataloader)-1)):
                with torch.no_grad():
                    generated_inputs = generator_net(fixed_noise2d).detach()
                    # generated_inputs = generator_net(fixed_noise3d).detach()
                    generated_inputs = torch.where(anim_mask[:4], torch_ones, generated_inputs)
                    # 2d
                    generated_inputs = generated_inputs.unsqueeze(0).cpu().numpy().transpose(0, 1, 3, 4, 2) 
                    # 3d
                    # generated_inputs = generated_inputs.reshape(-1, 108, 64, 64).unsqueeze(0).cpu().numpy().transpose(0, 1, 3, 4, 2) 
                    generated_img_cubes = generated_inputs if generated_img_cubes is None else np.concatenate((generated_img_cubes, generated_inputs), axis=0)

            train_iters += 1
        
    return generator_losses, discriminator_losses, generated_img_cubes
