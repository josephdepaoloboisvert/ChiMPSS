#!/usr/bin/env python
"""Run an Autodock Vina docking job."""

import argparse
import os


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def main():
    parser = argparse.ArgumentParser(
        description='Run an Autodock Vina docking job.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-r', '--receptor_path', type=str, required=True,
                        help='Path to .pdbqt file of receptor')
    parser.add_argument('-l', '--ligand_path', type=str, required=True,
                        help='Path to .pdbqt file of ligand')
    parser.add_argument('-ol', '--ligand_out_dir', type=str, default='./',
                        help='Directory to store docked poses')
    parser.add_argument('--config_dir', type=str, default='./',
                        help='Directory to store Autodock Vina configuration files')
    parser.add_argument('--box_center', type=list_of_ints, default=[0, 0, 0],
                        help='Box-center coordinates: x,y,z')
    parser.add_argument('--box_dim', type=list_of_ints, default=[15, 15, 15],
                        help='Box dimensions in Angstrom: x,y,z')
    parser.add_argument('--n_poses', type=int, default=20,
                        help='Number of docked poses')
    parser.add_argument('--exhaustiveness', type=int, default=8,
                        help='Level of exhaustiveness')
    parser.add_argument('--min_rmsd', type=float, default=1.0,
                        help='Minimum RMSD between saved poses')
    parser.add_argument('--compare', nargs=3, metavar=('REF_PDB', 'CHAIN', 'SELECTION'),
                        help='Reference PDB, chain ID, and MDAnalysis selection string '
                             'for comparison. E.g.: --compare /path/ref.pdb R "resname LIG"')
    args = parser.parse_args()

    if not os.path.exists(args.receptor_path):
        raise FileNotFoundError(f"Receptor file not found: {args.receptor_path}")
    if not os.path.exists(args.ligand_path):
        raise FileNotFoundError(f"Ligand file not found: {args.ligand_path}")

    from Docking.Docking import Docking

    docking = Docking(
        receptor_path=args.receptor_path,
        ligand_path=args.ligand_path,
        config_dir=args.config_dir,
    )
    docking.set_box(
        box_center=args.box_center,
        box_dim=args.box_dim,
    )
    docking.dock(
        lig_out_dir=args.ligand_out_dir,
        n_poses=args.n_poses,
        exhaustiveness=args.exhaustiveness,
        min_rmsd=args.min_rmsd,
    )

    if args.compare is not None:
        ref_pdb, ref_chainid, ref_lig_sele_str = args.compare
        docking.compare(
            ref_pdb=ref_pdb,
            ref_chainid=ref_chainid,
            ref_lig_sele_str=ref_lig_sele_str,
        )


if __name__ == '__main__':
    main()
