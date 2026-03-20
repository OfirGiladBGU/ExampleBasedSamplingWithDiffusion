"""Train the ControlNet adapter for grayscale-conditioned stipple generation.

Usage (from project root):
    python control_v1/train_control.py \
        --config  config/GBN/config.json \
        --ckpt    config/GBN/model.ckpt \
        --cond    /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/source \
        --offsets /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/processed_offsets \
        --epochs  100 \
        --batch_size 16 \
        --lr 1e-4 \
        --out control_v1/control_out
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.Config import ParseSampleConfig
from control_v1.ControlNet import ControlNet
from control_v1.StippleDataset import StippleDataset

WANDB_ENV = "/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/.env"
WANDB_ACTIVE = False


def load_wandb_key():
    if os.path.exists(WANDB_ENV):
        with open(WANDB_ENV) as f:
            for line in f:
                if line.strip().startswith("WANDB_API_KEY"):
                    key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["WANDB_API_KEY"] = key
                    return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/GBN/config.json",
                        help="Path to config.json")
    parser.add_argument("--ckpt", default="config/GBN/model.ckpt",
                        help="Path to model.ckpt")
    parser.add_argument("--cond",
                        default="/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/source",
                        help="Dir of grayscale condition images")
    parser.add_argument("--offsets",
                        default="/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/processed_offsets",
                        help="Dir of .npy offset files from prepare_data.py")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="control_v1/control_out",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # ---- wandb ----
    use_wandb = WANDB_ACTIVE
    if use_wandb:
        try:
            import wandb
            load_wandb_key()
            run_name = datetime.now().strftime("v1-train-%Y%m%d-%H%M%S")
            wandb.init(
                project="Stipple-ControlNet",
                name=run_name,
                config=vars(args),
            )
            print(f"wandb run name: {run_name}")
        except ImportError:
            print("wandb not installed, logging disabled")
            use_wandb = False

    # ---- Load pretrained diffusion model ----
    diffusion = ParseSampleConfig(args.config)
    diffusion.load_state_dict(torch.load(args.ckpt, map_location="cpu")["diffu"])
    diffusion.to(device)
    diffusion.eval()

    denoiser = diffusion.model
    num_timesteps = diffusion.num_timesteps

    for p in denoiser.parameters():
        p.requires_grad = False

    # ---- Build ControlNet ----
    control_net = ControlNet(denoiser).to(device)
    control_net.train()

    trainable_params = sum(p.numel() for p in control_net.parameters()
                           if p.requires_grad)
    total_locked = sum(p.numel() for p in denoiser.parameters())
    print(f"Trainable ControlNet params : {trainable_params:,}")
    print(f"Frozen denoiser params      : {total_locked:,}")

    # ---- Dataset & DataLoader ----
    dataset = StippleDataset(args.cond, args.offsets)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(control_net.parameters(), lr=args.lr)

    # ---- Training loop ----
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for cond, x_0 in dataloader:
            cond = cond.to(device)
            x_0 = x_0.to(device)

            t = torch.randint(0, num_timesteps, (x_0.shape[0],),
                              device=device)
            noise = torch.randn_like(x_0)
            x_t = diffusion.q_sample(x_0, t, noise)

            controls = control_net(x_t, t, cond)
            noise_pred = denoiser(x_t, t, controls=controls)

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(control_net.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            if use_wandb:
                wandb.log({"loss/step": loss.item()}, step=global_step)
            global_step += 1

        avg_loss = epoch_loss / max(len(dataloader), 1)
        if use_wandb:
            wandb.log({"loss/epoch": avg_loss, "epoch": epoch}, step=global_step)
        print(f"Epoch {epoch:>4d}  |  avg loss = {avg_loss:.6f}")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(args.out, f"controlnet_ep{epoch+1}.pt")
            torch.save({
                "control_net": control_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            }, save_path)
            print(f"  -> saved {save_path}")

    if use_wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
