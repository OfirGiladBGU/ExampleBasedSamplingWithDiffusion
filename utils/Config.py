import json
import torch
import os
import shutil
import torchvision
import numpy as np

DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def parse_indices(lst):
    indices = []
    for idx in lst:
        parts = idx.split(':')
        parts = [int(p) for p in parts]

        if len(parts) == 1:
            indices.append(parts[0])
        else:
            indices.extend(list(range(*parts)))
    return indices

def parse_model(model_params):
    from models.Denoiser import  DenoiserModel
    cond = model_params.get("cond", None)
    
    if cond is None:
        return DenoiserModel(**model_params), Ellipsis, []
    
    del model_params["cond"]

    if not cond["model"]:
        return DenoiserModel(**model_params), Ellipsis, []

    selects = parse_indices(cond["select"])
    masks = [parse_indices(m) for m in cond.get("masks", [])]
    
    if len(selects) == 0:
        return DenoiserModel(**model_params), Ellipsis, [[]]

    if len(masks) == 0:
        masks = [[]]

    if cond["model"] == "id":
        model = torch.nn.Identity()
    else:
        print("Unknown conditionnal model !")

    # Select subsequence for conditionning
    return DenoiserModel(**model_params, cond_features=cond.get("model_out", None), cond_emb_dim=cond.get("embd_dim", None), cond_model=model), selects, masks

def parse_train(train_params, select_cond, mask_cond):
    from data.Dataset import QMCDataset
    if "scales" not in train_params:
        train_params["scales"] = [".*"]

    dataset = QMCDataset(train_params["dataset"], samplers=train_params["samplers"], scales=train_params["scales"], cond_index=select_cond, mask_index=mask_cond)

    del train_params["dataset"]
    del train_params["scales"]
    return dataset, train_params

def parse_eval(path, eval_params):
    from utils.Eval import EvalPointset
    return EvalPointset(path, eval_params["freq"], eval_params["ts"])

def parse_diffusion(diffu_params, model, device=DEFAULT_DEVICE):
    from models.Diffusion import  DiffusionModel
    betas = np.linspace(
        diffu_params["betas"]["min"],
        diffu_params["betas"]["max"],
        diffu_params["betas"]["count"]
    )
    loss = diffu_params["loss"]
    return DiffusionModel(model, betas=betas, loss=loss, device=device)

def parse_flow(flow_params, model, device=DEFAULT_DEVICE):
    """Flow Matching process. Only the keys FM actually uses -- no betas, no schedule."""
    from models.FlowMatchingProcess import FlowMatchingModel
    return FlowMatchingModel(
        model,
        device=device,
        loss=flow_params.get("loss", "mse"),
        t_scale=flow_params.get("t_scale", 1000.0),
        eps_t=flow_params.get("eps_t", 1e-4),
        t_dist=flow_params.get("t_dist", "uniform"),
        t_logitnorm_m=flow_params.get("t_logitnorm_m", 0.0),
        t_logitnorm_s=flow_params.get("t_logitnorm_s", 1.0),
        steps=flow_params.get("steps", 50),
        ode_method=flow_params.get("ode_method", "euler"),
    )

def parse_process(config, model, device=DEFAULT_DEVICE):
    """Select the generative process from the config: exactly one of 'diffusion' | 'flow'."""
    has_diff, has_flow = "diffusion" in config, "flow" in config
    if has_diff and has_flow:
        raise ValueError("Config declares BOTH 'diffusion' and 'flow'; pick exactly one.")
    if has_flow:
        return parse_flow(config["flow"], model, device=device)
    if has_diff:
        return parse_diffusion(config["diffusion"], model, device=device)
    raise ValueError("Config must declare a process: either a 'diffusion' or a 'flow' block.")

def ParseTrainConfig(path):
    from utils.Trainer import Trainer
    with open(path) as config_file:
        config = json.load(config_file)
        
        assert "model" in config
        assert "diffusion" in config or "flow" in config
        assert "train" in config
        assert "eval" in config
        assert "path" in config

        model, s, m = parse_model(config["model"])
        diffu = parse_process(config, model)
        dataset, params = parse_train(config["train"], s, m)
        eval = parse_eval(config["path"], config["eval"])

    os.makedirs(config["path"], exist_ok=True)
    shutil.copy(path, os.path.join(config["path"], "config.json"))

    return Trainer(
        config["path"],
        model=model, 
        dataset=dataset, 
        diffusion=diffu, 
        train_params=params,
        eval=eval,
        device=DEFAULT_DEVICE
    )

def ParseSampleConfig(path, device=DEFAULT_DEVICE):
    with open(path) as config_file:
        config = json.load(config_file)
        
        assert "model" in config
        assert "diffusion" in config or "flow" in config

        model, s, m = parse_model(config["model"])
        diffu = parse_process(config, model, device=device)

        return diffu
