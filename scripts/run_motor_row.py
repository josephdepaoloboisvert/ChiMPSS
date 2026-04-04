import os, sys, glob, argparse
from chimpss.motor_row.motor_row import MotorRow
parser = argparse.ArgumentParser()
parser.add_argument('pdb_file', type=str, help='Path to PDB File with starting coordinates (This shipped from Bridgeport)')
parser.add_argument('system_xml', type=str, help='Path to the XML File with the serialized system (This shipped from Bridgeport)')
parser.add_argument('output_dir', type=str, help='Directory to store output')
parser.add_argument('--lig_resname', default='UNK', type=str, help='Three letter resname for the ligand (put in double quotes if starts with number)')
args = parser.parse_args()
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)
MotorRow(pdb_file=args.pdb_file, system_xml=args.system_xml, working_directory=args.output_dir, lig_resname=args.lig_resname).main(args.pdb_file)

