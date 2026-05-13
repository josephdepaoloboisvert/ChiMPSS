"""
CLI entry point for MotorRow equilibration.

Console script: chimpss-motorrow
"""

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(
        description='Run MotorRow 5-step equilibration protocol.')
    p.add_argument('pdb_file',   type=str,
                   help='Path to PDB file with starting coordinates (output from Bridgeport)')
    p.add_argument('system_xml', type=str,
                   help='Path to XML file with the serialized system (output from Bridgeport)')
    p.add_argument('output_dir', type=str,
                   help='Directory to store output')
    p.add_argument('--lig_resname', default='UNK', type=str,
                   help='Three-letter residue name for the ligand (default: UNK)')
    p.add_argument('--protein-name', default=None, type=str,
                   help='Protein name for output file naming (no underscores or periods). '
                        'When combined with --ligand-name, final outputs are named '
                        'PROTEIN_LIGAND.equil.pdb / PROTEIN_LIGAND.state.xml.')
    p.add_argument('--ligand-name', default=None, type=str,
                   help='Ligand name for output file naming (no underscores or periods).')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    from chimpss.motorrow import MotorRow
    MotorRow(pdb_file=args.pdb_file,
             system_xml=args.system_xml,
             working_directory=args.output_dir,
             lig_resname=args.lig_resname,
             protein_name=args.protein_name,
             ligand_name=args.ligand_name).main(args.pdb_file)


if __name__ == '__main__':
    main()
