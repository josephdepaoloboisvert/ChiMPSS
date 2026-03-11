#RUN SOME FUCKING HMR BIG TIMESTEP ENERGY BOIIII

#MY TESTING SYSTEM
# input_pdb='/expanse/lustre/projects/uil133/josephdb/SimulationData/AM630_fg_MotorRow/Step_5.pdb'
# input_system='/expanse/lustre/projects/uil133/josephdb/ChiMPSS/for_Jo/AM630_FG_HMR_sys.xml'
# input_state='/expanse/lustre/projects/uil133/josephdb/SimulationData/AM630_fg_MotorRow/Step_5.xml'
# output_dir='/expanse/lustre/projects/uil133/josephdb/SimulationData/AM630_HMR_3.5_FultonMarket/'

import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument('input_pdb', type=str, help='Path to PDB File with starting coordinates')
parser.add_argument('input_system', type=str, help='Path to the XML File with the serialized system')
parser.add_argument('output_dir', type=str, help='Directory to store output')
parser.add_argument('--input_state', default=None, type=str, help='Path to XML File of a recently written state')
parser.add_argument('--T_min', default=310, type=float, help='Lowest Temperature on the Temperature ladder')
parser.add_argument('--T_max', default=350, type=float, help='Highest Temperature on the Temperature ladder')
parser.add_argument('--n_replicates', default=40, type=int, help='Number of states (temperatures) for Replica Exchange')
parser.add_argument('--iter_length', default=0.010, type=float, help='Time (nanoseconds) between swaps')
parser.add_argument('--timestep', default=3.5, type=float, help='Time (femtoseconds) for the integration steps')
parser.add_argument('--sim_length', default=35, type=int, help='Time (nanoseconds) between checkpointed folder creations')
parser.add_argument('--total_sim_time', default=3500, type=int, help='Time (nanoseconds) of desired aggregate simulation (across replicates)')

args = parser.parse_args()

from FultonMarket.FultonMarket import FultonMarket as FM

market = FM(input_pdb=args.input_pdb,
            input_system=args.input_system,
            input_state=args.input_state, 
            T_min=args.T_min,
            T_max=args.T_max,
            n_replicates=args.n_replicates)

if not os.path.isdir(args.output_dir):
    print(f"Creating an Output Directory at: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

market.run(iter_length=args.iter_length,
           dt=args.timestep,
           sim_length=args.sim_length,
           total_sim_time=args.total_sim_time,
           output_dir=args.output_dir)


