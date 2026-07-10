import os
import time

import torch
import numpy as np

from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from ema_pytorch import EMA

from models.FlowMatchingProcess import FlowMatchingModel

def cycle(dataset):
    while True:
        for data in dataset:
            yield data

class Trainer:
    def __init__(self,
        path,
        model,
        dataset,
        diffusion,
        train_params,
        eval,
        device = torch.device('cuda')
    ):
        self.device = device
        self.train_params = train_params
        self.eval = eval

        self.dataset = cycle(DataLoader(
            dataset, 
            batch_size = train_params["batch_size"], 
            shuffle = True, 
            pin_memory = True,
            num_workers = 10,
            prefetch_factor = 1
        ))
        # Ask the process, so a Flow-Matching process (no betas) also works.
        self.num_timesteps = diffusion.num_timesteps

        self.model = model.to(device)
        self.diffusion = diffusion.to(device)
        self.ema = EMA(
            self.diffusion,
            beta        = train_params["ema_decay"],
            update_every= train_params["ema_update_every"]
        ).to(device)

        # The EMA weights were historically computed every step and then thrown away:
        # Eval.compute saves and samples the RAW module, and Trainer.save (the only writer
        # of the 'ema' key) is never called. Turning EMA on changes what model.ckpt holds,
        # and ~20 downstream scripts load that file as the frozen DDPM base via ["diffu"] --
        # so it stays OFF for a 'diffusion' process and ON for a 'flow' process, where no
        # existing checkpoint depends on the old behaviour. Override via train.use_ema_for_eval.
        self.use_ema_for_eval = bool(train_params.get(
            "use_ema_for_eval", isinstance(diffusion, FlowMatchingModel)
        ))

        self.optim = torch.optim.Adam(self.model.parameters(), lr=train_params["lr"])
        self.writer = SummaryWriter(path)
        # torch.backends.cudnn.benchmark = True

    def eval_model(self):
        """The weights used for eval sampling AND written to model.ckpt.

        EMA weights when use_ema_for_eval, else the raw module. Returning the same object
        the checkpoint is built from is what keeps 'the panel you looked at' and 'the
        weights you shipped' the same thing.
        """
        if not self.use_ema_for_eval:
            return self.diffusion

        ema_model = getattr(self.ema, "ema_model", None)
        if ema_model is None:
            raise RuntimeError(
                "use_ema_for_eval=True but ema_pytorch.EMA exposes no `.ema_model`; "
                "check the installed ema_pytorch version."
            )
        return ema_model

    def save(self, path):
        data = {
            # 'model': self.model.state_dict(),
            # 'diffu' (not 'diff') is the key every loader in this repo reads.
            'diffu': self.eval_model().state_dict(), # Includes model
            'raw': self.diffusion.state_dict(),
            'optim': self.optim.state_dict(),
            'ema': self.ema.state_dict()
        }
        torch.save(data, path)

    @torch.no_grad()
    def sample(self, noise, cond=None):
        self.model.eval()
        self.diffusion.eval()
        self.ema.eval()

        model = self.eval_model()
        model.eval()
        data = model.p_sample_loop(None, noise, cond).detach().cpu().numpy()

        self.ema.train()
        self.diffusion.train()
        self.model.train()

        return data

    def sample_with_grad(self, noise, cond=None):
        # No TQDM, No Sampling
        samples = self.diffusion.p_sample_loop(None, noise, cond, False, False)
        return samples
    
    def train(self, it, withtqdm=True, to=1e8):
        its = range(it)
        if withtqdm:
            from tqdm import tqdm
            its = tqdm(its)

        tinit = time.time()
        end = False

        for i in its:
            data = next(self.dataset)
            # Loss + optim step
            self.optim.zero_grad()

            # Data contains a mapping dimension to samples 
            loss = 0
            
            # Same timestep for evey scale
            bs = data[list(data.keys())[0]]["data"].shape[0]
            # The process decides what a "timestep" is: integer for DDPM, continuous for FM.
            ts = self.diffusion.sample_timesteps(bs).to(self.device)
            
            for scale, scale_data in data.items():
                d = scale_data["data"].to(self.device)
                c = scale_data["prop"].to(self.device)
                
                scale_loss = self.diffusion(d, ts, x_cond = c)
                self.writer.add_scalar(f"Loss/{scale}", scale_loss / int(scale), i)
                
                loss += scale_loss

            self.writer.add_scalar(f"Loss/all", loss, i)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            
            self.optim.step()

            # Exponential moving average pass
            self.ema.update()

            runtime = (time.time() - tinit) / 60
            end = runtime > to
            
            with torch.no_grad():
                self.diffusion.eval()
                model = self.eval_model()
                model.eval()
                self.eval.compute(
                    it=i, writer=self.writer, model=model, data=data,
                    force=(end or (i == it - 1)),
                    raw_model=self.diffusion,
                )
                self.diffusion.train()

            if end:
                break