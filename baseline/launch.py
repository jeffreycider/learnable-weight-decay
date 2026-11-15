"""Pin CPU inter-op concurrency before the PR323 training entrypoint.

The original seed0 control reported20. Pinning this value prevents host-dependent
defaults from becoming an unrecorded difference between subsequent paired runs.
The training environment record reports the actual active value.
"""
import torch


def main():
    torch.set_num_interop_threads(20)
    from .train import main as train
    train()


if __name__ == '__main__':
    main()
