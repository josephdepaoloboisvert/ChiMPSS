#!/usr/bin/env python

"""
Standalone MPI wrapper for FultonMarket simulations
Usage: mpirun -np 4 python mpi_fultonmarket_wrapper.py [RUN_FULTONMARKET.py arguments]
This script requires your original RUN_FULTONMARKET.py to exist in the same directory

Example:
mpirun -np 4 python RUN_FULTONMARKET_MPI.py /path/to/output 8 /path/to/replica_exchange 0 -t 1000 -s 25 -n 120
"""

import os
import sys

def setup_mpi_gpu():
    """Setup MPI and GPU assignment for FultonMarket"""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Get available GPUs from environment
        available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
        gpu_id = available_gpus[rank % len(available_gpus)]
        
        # CRITICAL: Set GPU for this process and hide all others
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        # Status output
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

def verify_gpu_isolation():
    """Verify that each process only sees its assigned GPU"""
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0
    
    gpu_env = os.environ.get('CUDA_VISIBLE_DEVICES', 'None')
    print(f"Rank {rank}: CUDA_VISIBLE_DEVICES = {gpu_env}")
    
    # Test with OpenMM if available
    try:
        from openmm import Platform
        cuda_available = False
        for i in range(Platform.getNumPlatforms()):
            if Platform.getPlatform(i).getName() == 'CUDA':
                cuda_available = True
                print(f"Rank {rank}: OpenMM CUDA platform available")
                break
        
        if not cuda_available:
            print(f"Rank {rank}: OpenMM CUDA platform not available")
            
    except ImportError:
        print(f"Rank {rank}: OpenMM not yet imported")
    
    # Use nvidia-ml-py if available for verification
    try:
        import pynvml
        pynvml.nvmlInit()
        total_gpus = pynvml.nvmlDeviceGetCount()
        visible_gpus = len(gpu_env.split(',')) if gpu_env != 'None' else 0
        print(f"Rank {rank}: System has {total_gpus} total GPUs, this process should see {visible_gpus}")
        
    except ImportError:
        print(f"Rank {rank}: pynvml not available for GPU verification")
    except Exception as e:
        print(f"Rank {rank}: GPU verification failed: {e}")
    
    return True

def execute_run_fultonmarket():
    """Execute the original RUN_FULTONMARKET.py script"""
    try:
        # Check if RUN_FULTONMARKET.py exists
        if not os.path.exists('RUN_FULTONMARKET.py'):
            raise FileNotFoundError("RUN_FULTONMARKET.py not found in current directory")
        
        # Get current rank for output control
        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
            if rank == 0:
                print("Executing RUN_FULTONMARKET.py with MPI GPU assignments...")
        except ImportError:
            print("Executing RUN_FULTONMARKET.py in single process mode...")
        
        # Execute the original script using runpy to preserve environment
        # Pass through all command line arguments
        original_argv = sys.argv[1:]  # Get arguments passed to wrapper
        sys.argv = ['RUN_FULTONMARKET.py'] + original_argv  # Set up argv for the script
        
        import runpy
        runpy.run_path('RUN_FULTONMARKET.py', run_name='__main__')
        
    except Exception as e:
        print(f"Error executing RUN_FULTONMARKET.py: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Main wrapper function"""
    print("=== FultonMarket MPI Wrapper ===")
    
    # Setup MPI and GPU assignment FIRST
    rank, size, gpu_id = setup_mpi_gpu()
    
    # Verify GPU isolation worked
    verify_gpu_isolation()
    
    # Brief check for script existence
    if not os.path.exists('RUN_FULTONMARKET.py'):
        print("ERROR: RUN_FULTONMARKET.py not found in current directory")
        sys.exit(1)
    
    if rank == 0:
        print("Starting FultonMarket simulation with MPI GPU assignments...")
        print(f"Command line arguments: {sys.argv[1:]}")
    
    # Execute the original RUN_FULTONMARKET.py script
    execute_run_fultonmarket()
    
    # Final status
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            print("=== FultonMarket simulation completed ===")
    except ImportError:
        print("=== FultonMarket simulation completed ===")

if __name__ == "__main__":
    main()
