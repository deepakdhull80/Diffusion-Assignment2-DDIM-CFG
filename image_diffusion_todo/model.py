from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scheduler import DDPMScheduler, DDIMScheduler

class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    def get_loss(self, x0, class_label=None, noise=None):
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute noise matching loss.
        B = x0.shape[0]
        timestep = self.var_scheduler.uniform_sample_t(B, self.device)        
        if noise is None:
            noise = torch.randn_like(x0).to(x0.device)
        x_t, noise = self.var_scheduler.add_noise(x0, timestep, noise)
        noise_p = self.network(x_t, timestep, class_label=class_label)
        
        loss = F.mse_loss(noise, noise_p, reduction='mean')
        ######################
        return loss
    
    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 0.0,
    ):
        x_T = torch.randn([batch_size, 3, self.image_resolution, self.image_resolution]).to(self.device)

        do_classifier_free_guidance = guidance_scale > 0.0

        if do_classifier_free_guidance:

            ######## TODO ########
            # Assignment 2-3. Implement the classifier-free guidance.
            # Specifically, given a tensor of shape (batch_size,) containing class labels,
            # create a tensor of shape (2*batch_size,) where the first half is filled with zeros (i.e., null condition).
            assert class_label is not None
            assert len(class_label) == batch_size, f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"
            if not isinstance(class_label, torch.Tensor):
                class_label = torch.tensor(class_label).to(self.device)
            null_condition = torch.zeros_like(class_label)
            class_label = torch.cat([null_condition, class_label], dim=0)
            #######################

        traj = [x_T]
        for t in tqdm(self.var_scheduler.timesteps):
            x_t = traj[-1]
            if do_classifier_free_guidance:
                ######## TODO ########
                # Assignment 2. Implement the classifier-free guidance.
                # Get conditional prediction (with class_label)
                eps_cond = self.network(
                    x_t,
                    timestep=t.to(self.device),
                    class_label=class_label[batch_size:],
                )
                
                # Get unconditional prediction (null condition)
                eps_uncond = self.network(
                    x_t,
                    timestep=t.to(self.device),
                    class_label=class_label[:batch_size],
                )
                
                # Apply guidance formula from Algorithm 2
                noise_pred = (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond
                # noise_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                #######################
            else:
                noise_pred = self.network(
                    x_t,
                    timestep=t.to(self.device),
                    class_label=class_label,
                )

            x_t_prev = self.var_scheduler.step(x_t, t, noise_pred)

            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]
    
    @torch.no_grad()
    def ddim_sample(
        self,
        batch_size,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 0.0,
        eta=0.0,
        num_inference_timesteps=50
    ):
        x_T = torch.randn([batch_size, 3, self.image_resolution, self.image_resolution]).to(self.device)

        do_classifier_free_guidance = guidance_scale > 0.0

        if do_classifier_free_guidance:

            ######## TODO ########
            # Assignment 2-3. Implement the classifier-free guidance.
            # Specifically, given a tensor of shape (batch_size,) containing class labels,
            # create a tensor of shape (2*batch_size,) where the first half is filled with zeros (i.e., null condition).
            assert class_label is not None
            assert len(class_label) == batch_size, f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"
            if not isinstance(class_label, torch.Tensor):
                class_label = torch.tensor(class_label).to(self.device)
            null_condition = torch.zeros_like(class_label)
            class_label = torch.cat([null_condition, class_label], dim=0)
            #######################

        traj = [x_T]
        
        step_ratio = self.var_scheduler.num_train_timesteps // num_inference_timesteps
        timesteps = (
            (np.arange(0, num_inference_timesteps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        timesteps = torch.from_numpy(timesteps)
        timesteps = timesteps.to(self.device)
        prev_timesteps = timesteps - step_ratio
        
        for t, t_prev in tqdm(zip(timesteps, prev_timesteps), total=timesteps.shape[0]):
            x_t = traj[-1]
            if do_classifier_free_guidance:
                ######## TODO ########
                # Assignment 2. Implement the classifier-free guidance.
                # Get conditional prediction (with class_label)
                eps_cond = self.network(
                    x_t,
                    timestep=t.to(self.device),
                    class_label=class_label[batch_size:],
                )
                
                # Get unconditional prediction (null condition)
                eps_uncond = self.network(
                    x_t,
                    timestep=t.to(self.device),
                    class_label=class_label[:batch_size],
                )
                
                # Apply guidance formula from Algorithm 2
                noise_pred = (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond
                # noise_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                #######################
            else:
                noise_pred = self.network(
                    x_t,
                    timestep=t.to(self.device),
                    class_label=class_label,
                )
            
            x_t_prev = self.var_scheduler.step(x_t, t, t_prev=t_prev, eps_theta=noise_pred, eta=eta)
            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
            } 
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
