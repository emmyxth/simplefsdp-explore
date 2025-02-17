#!/usr/bin/env python3

import os
import torch
import traceback
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

# Import your SimpleFSDP_data_parallel wrapper from wherever it's defined
from frontend_simplefsdp import SimpleFSDP_data_parallel
from torch.distributed.tensor import distribute_tensor, Replicate

########################################################################
# 1. Define the same Net class
########################################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=4)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


########################################################################
# 2. Define the per-process function that:
#    - Initializes distributed, picks correct GPU
#    - Builds device mesh (2 GPUs)
#    - Wraps model with SimpleFSDP_data_parallel
#    - Compiles model with torch.compile
#    - Runs a forward pass on dummy input
########################################################################
def fsdp_demo(rank, world_size):
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # 2a. Initialize the process group with NCCL (for GPUs)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        # 2b. Pin the current process to its GPU
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        # 2c. Instantiate the model on this GPU
        model = Net().to(device)
        model.eval()

        # 2d. Build a device mesh with shape (2,) for 2 GPUs
        #     In a 2-GPU setup, rank 0 uses 'cuda:0', rank 1 uses 'cuda:1'.
        #     init_device_mesh will gather all visible GPUs, so (2,) is fine.
        mesh = init_device_mesh(device_type="cuda", mesh_shape=(2,))

        mp_policy = MixedPrecisionPolicy()

        # 2e. Wrap the model with SimpleFSDP_data_parallel
        fsdp_model = SimpleFSDP_data_parallel(
            model=model,
            device_mesh=mesh,
            mode="replicate",
            reshard_after_forward=True,
            mp_policy=mp_policy
        )

        # 2f. Compile the FSDP-wrapped model
        # compiled_model = torch.compile(fsdp_model)
        compiled_model = fsdp_model


        # 2g. Dummy input of shape (batch=1, channel=1, 28x28)
        dummy_input = torch.randn(1, 1, 32, 32, device=device)
        dummy_input_dt = distribute_tensor(
            dummy_input,
            device_mesh=mesh,         # The same mesh you used for the model
            placements=[Replicate()]  # Typically replicate the input across each rank
        )

        # 2h. Forward pass
        with torch.no_grad():
            output = compiled_model(dummy_input_dt)

        # 2i. Print result only on rank 0
        if rank == 0:
            print(f"[Rank 0] Output shape: {output.shape}")
            print(f"[Rank 0] Output:\n{output}")

        # 2j. Cleanup
        dist.destroy_process_group()
    except Exception as e:
        # print(e)
        traceback.print_exc()
        dist.destroy_process_group()


########################################################################
# 3. Main entry point: spawn 2 processes (one per GPU)
########################################################################
if __name__ == "__main__":
    print("torch.cuda.device_count():", torch.cuda.device_count())
    world_size = 2  # Make sure you have exactly 2 GPUs visible
    mp.spawn(fsdp_demo, args=(world_size,), nprocs=world_size, join=True)