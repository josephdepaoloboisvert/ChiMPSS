"""
MPI wrapper for FultonMarket simulations.

Console script: chimpss-fultonmarket-mpi

Usage:
  mpiexec.hydra -np 4 chimpss-fultonmarket-mpi <chimpss-fultonmarket args>
"""

import os
import sys


def setup_mpi_gpu():
    """Set up MPI rank and assign an isolated GPU to this process."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
        gpu_id = available_gpus[rank % len(available_gpus)]

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

        if rank == 0:
            print(f"FultonMarket MPI Wrapper: {size} processes launched")
            print(f"Original GPUs available: {available_gpus}")
        print(f"Rank {rank}: CUDA_VISIBLE_DEVICES={gpu_id}")

        return rank, size, gpu_id

    except ImportError:
        print("Warning: MPI not available, running single process")
        gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        return 0, 1, gpu_id


def main():
    print("=== FultonMarket MPI Wrapper ===")
    rank, size, gpu_id = setup_mpi_gpu()

    if rank == 0:
        print(f"Command line arguments: {sys.argv[1:]}")
        print("Starting FultonMarket simulation with MPI GPU assignments...")

    # Delegate to the single-process CLI main(), which reads sys.argv directly.
    from chimpss.cli.fultonmarket import main as fm_main
    fm_main()

    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("=== FultonMarket simulation completed ===")
    except ImportError:
        print("=== FultonMarket simulation completed ===")


if __name__ == '__main__':
    main()
