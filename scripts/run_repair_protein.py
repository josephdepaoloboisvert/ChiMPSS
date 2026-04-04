#!/usr/bin/env python
"""
Repair missing residues in PDB files using MODELLER.

All target proteins must share the same template sequence.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description='Repair missing residues in PDB files using MODELLER.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-i', '--input_dir', required=True,
                        help='Directory containing input .pdb files')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Directory for repaired .pdb files')
    parser.add_argument('-f', '--fasta', required=True,
                        help='Path to .fasta file used as the template sequence')
    parser.add_argument('--tails', action='store_true', default=False,
                        help='Add tail residues to N and C termini')
    args = parser.parse_args()

    from RepairProtein.repair_protein import RepairProtein

    working_dir = os.getcwd()
    int_dir = os.path.join(working_dir, 'modeller_intermediates')
    if not os.path.exists(int_dir):
        os.mkdir(int_dir)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for pdb in os.listdir(args.input_dir):
        rp = RepairProtein(
            pdb_fn=os.path.join(args.input_dir, pdb),
            fasta_fn=args.fasta,
            working_dir=int_dir,
        )
        rp.run(
            pdb_out_fn=os.path.join(args.output_dir, pdb),
            tails=args.tails,
        )


if __name__ == '__main__':
    main()
