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
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)


# Import your SimpleFSDP_data_parallel wrapper from wherever it's defined
from frontend_simplefsdp import SimpleFSDP_data_parallel, enable_active_parametrization
from torch.distributed.tensor import distribute_tensor

torch._logging.set_logs(graph_code=True)  # Logging set up


# --- Patch F.conv2d ---
_original_conv2d = F.conv2d


def patched_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # If the input is a DTensor, we assume its _spec contains the mesh and placements
    if isinstance(input, DTensor):
        mesh = input._spec.mesh
        placements = input._spec.placements
        if not isinstance(weight, DTensor):
            weight = DTensor.from_local(weight, mesh, placements)
        if bias is not None and not isinstance(bias, DTensor):
            bias = DTensor.from_local(bias, mesh, placements)
    return _original_conv2d(input, weight, bias, stride, padding, dilation, groups)


F.conv2d = patched_conv2d

# --- Patch F.linear ---
_original_linear = F.linear


def patched_linear(input, weight, bias=None):
    if isinstance(input, DTensor):
        mesh = input._spec.mesh
        placements = input._spec.placements
        if not isinstance(weight, DTensor):
            weight = DTensor.from_local(weight, mesh, placements)
        if bias is not None and not isinstance(bias, DTensor):
            bias = DTensor.from_local(bias, mesh, placements)
    return _original_linear(input, weight, bias)


F.linear = patched_linear


########################################################################
# 1. Define the same Net class
########################################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Use kernel_size=2 and stride=2, with no padding, which meets the tensor parallel requirements.
        # For a 28x28 input, the first conv produces an output of size: (28 - 2) / 2 + 1 = 14.
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=2, stride=2, padding=0
        )
        # Second conv: input size 14 -> (14 - 2) / 2 + 1 = 7.
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=2, stride=2, padding=0
        )
        # After conv2, the feature map is 32 channels of 7x7, so flattened size = 32*7*7 = 1568.
        self.fc = nn.Linear(1568, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


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
            mp_policy=mp_policy,
        )
        # 2f. Compile the FSDP-wrapped model
        compiled_model = torch.compile(fsdp_model)
        # compiled_model = fsdp_model

        # 2g. Dummy input of shape (batch=1, channel=1, 28x28)
        dummy_input = torch.randn(1, 1, 28, 28, device=device)
        dummy_input_dt = distribute_tensor(
            dummy_input,
            device_mesh=mesh,  # The same mesh you used for the model
            placements=[Replicate()],  # Typically replicate the input across each rank
        )
        with enable_active_parametrization():
            output = compiled_model(dummy_input_dt)
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
