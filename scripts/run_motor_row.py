#!/usr/bin/env python
"""Run a MotorRow membrane protein equilibration."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description='Run a MotorRow membrane protein equilibration simulation.'
    )
    parser.add_argument('pdb_file',   type=str, help='Path to PDB file with starting coordinates (from Bridgeport)')
    parser.add_argument('system_xml', type=str, help='Path to XML file with the serialized system (from Bridgeport)')
    parser.add_argument('output_dir', type=str, help='Directory to store output')
    parser.add_argument('--lig_resname', default='UNK', type=str,
                        help='Three-letter residue name for the ligand '
                             '(use double quotes if name starts with a digit). Default UNK.')
    args = parser.parse_args()

    from chimpss.motor_row.motor_row import MotorRow

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    MotorRow(
        pdb_file=args.pdb_file,
        system_xml=args.system_xml,
        working_directory=args.output_dir,
        lig_resname=args.lig_resname,
    ).main(args.pdb_file)


if __name__ == '__main__':
    main()
