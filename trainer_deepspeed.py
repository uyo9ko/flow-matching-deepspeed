import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import datasets
import PIL
from einops import rearrange
from accelerate import Accelerator, DataLoaderConfiguration

from classes import IMAGENET2012_CLASSES
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
from tqdm import tqdm
import wandb
from typing import *
import yaml
import math

from networks.unet import Unet
from networks.flux_ae import ae_transform, load_ae
from flows.rectified_flow import RectifiedFlow

def collate_fn(batch, image_size=(256, 256)):
    if not isinstance(image_size, (list, tuple)):
        image_size = (image_size, image_size)
        
    transform = Compose([
        Resize(image_size[0], antialias=True),  # Resize shortest edge while maintaining aspect ratio
        CenterCrop(image_size),  # Crop to desired size from center
    ])
    
    images = [transform(item['jpeg'].convert('RGB')) for item in batch]
    images = [ae_transform(img) for img in images]
    labels = [item['__key__'].split('_')[0] for item in batch  ]
    label_indices = [list(IMAGENET2012_CLASSES.keys()).index(label) for label in labels]
    return {
        'images': torch.stack(images),
        'labels': torch.tensor(label_indices)
    }


def training_function(config, args):
    # Initialize accelerator
    dataloader_config = DataLoaderConfiguration(use_stateful_dataloader=args.use_stateful_dataloader)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_dir=args.wandb_project_dir,
        dataloader_config=dataloader_config,
    )


    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    image_size = config["image_size"]
    model_dim = config["model_dim"]
    max_train_steps = config["max_train_steps"]
    if not isinstance(image_size, (list, tuple)):
        image_size = (image_size, image_size)
    checkpointing_steps = int(args.checkpointing_steps)
    sampling_steps = int(args.sampling_steps)

    # Parse out whether we are saving every epoch or after a certain number of batches

    # We need to initialize the trackers we use, and also store our configuration
    run = 'rectified_flow'
    accelerator.init_trackers(run, config)

    # Set the seed before splitting the data.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = datasets.load_dataset(path=args.data_dir, split='train', streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        collate_fn=lambda batch: collate_fn(batch, image_size=image_size)  # Pass image_size to collate_fn
    )

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    net = Unet(channels = 16, dim = model_dim, num_class=1000)

    ae_ckpt_path = '/data_training/larry/code/dit/flow-matching-deepspeed/ae_weights/ae.safetensors'
    ae = load_ae(device=accelerator.device, ckpt_path=ae_ckpt_path)
    ae.requires_grad_(False)

    param_count = sum(p.numel() for p in net.parameters())
    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    net = net.to(accelerator.device)
    flow_model = RectifiedFlow(net, accelerator.device)

    # Instantiate optimizer
    optimizer = torch.optim.Adam(params=flow_model.parameters(), lr=lr) 
    # Instantiate learning rate scheduler
    lr_scheduler = None

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    flow_model, optimizer, dataloader = accelerator.prepare(
        flow_model, optimizer, dataloader     
    )


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    data_set_len = 1_281_167
    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the starting epoch so files are named properly
    starting_epoch = 0
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(data_set_len / (batch_size * config["gradient_accumulation_steps"]))
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = batch_size * accelerator.num_processes * config["gradient_accumulation_steps"]


    if accelerator.is_main_process: 
        print("***** Running training *****")
        print(f"  Number of model parameters: {param_count:,} ({param_count/1e6:.2f}M)")
        print(f"  Num batches each epoch = {data_set_len // batch_size}")
        print(f"  Num Epochs = {num_train_epochs}")
        print(f"  Instantaneous batch size per device = {batch_size}")
        print(f"  Gradient Accumulation steps = {config['gradient_accumulation_steps']}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Total optimization steps = {max_train_steps}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]
        resume_step = int(training_difference.replace("step_", ""))
        starting_epoch = resume_step // data_set_len
        resume_step -= starting_epoch * data_set_len

    # Now we train the model
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=overall_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(starting_epoch, num_epochs):
        flow_model.train()
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            overall_step += resume_step
            progress_bar.update(resume_step)
        active_dataloader = dataloader
        # start training
        for batch in active_dataloader:
            with accelerator.accumulate(flow_model):
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                images = batch['images']
                labels = batch['labels']
                # batch = batch.to(accelerator.device)
                with torch.no_grad():
                    latents = ae.encode(images.to(torch.float32))
                latents = latents.to(weight_dtype)
                loss = flow_model(latents, labels)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                overall_step += 1

                if overall_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        output_dir = f"step_{overall_step}"
                        if args.ckpt_dir is not None:
                            output_dir = os.path.join(args.ckpt_dir, output_dir)
                        os.makedirs(output_dir, exist_ok=True)
                        accelerator.save_state(output_dir)

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=overall_step)

            if overall_step >= max_train_steps:
                break

            if accelerator.is_main_process:
                if overall_step % sampling_steps == 0:
                    flow_model.eval()
                    # Generate samples
                    labels = torch.randint(0, 1000, (4,)).to(accelerator.device)
                    samples = accelerator.unwrap_model(flow_model).sample(labels, batch_size=4, data_shape=latents.shape[1:])
                    with torch.no_grad():
                        samples = ae.decode(samples)
                    
                    # Convert and save each sample
                    images = []
                    for i, sample in enumerate(samples):
                        # Denormalize and convert to PIL image
                        # sample = (sample.cpu().clamp(-1, 1) + 1) / 2 * 255
                        # sample = sample.permute(1, 2, 0).numpy().astype(np.uint8)
                        sample = sample.clamp(-1, 1)
                        sample = rearrange(sample, "c h w -> h w c")
                        img = PIL.Image.fromarray((127.5 * (sample + 1.0)).cpu().byte().numpy())
                        images.append(img)
                        
                    # Log the image if tracking is enabled
                    accelerator.log({
                        'samples': [wandb.Image(image) for image in images]
                    }, step=overall_step)


    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--data_dir", required=True, help="The data folder on disk.")
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help="Whether the various states should be saved at the end of every n steps"
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,
        help="Optional number of sampling steps to perform.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        default=".",
        help="Optional save directory where all samples folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--use_stateful_dataloader",
        action="store_true",
        help="If the dataloader should be a resumable stateful dataloader.",
    )
    parser.add_argument(
        "--wandb_project_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs` and relevent project information",
    )
    args = parser.parse_args()
    config = {"lr": 1e-4,
              "num_epochs": 10,
              "seed": 42,
              "batch_size": 64,
              "gradient_accumulation_steps": 1,
              "num_workers": 4,
              "pin_memory": True,
              "image_size": 256,
              "model_dim": 128,
              "max_train_steps": 100000
              }
    training_function(config, args)


if __name__ == "__main__":
    main()