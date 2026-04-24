"""
Grid potential utilities: IO_Grid reader/writer plus OpenMM system helpers.

Migrated from grid_potentials.py (root). Module-level script code removed;
only the reusable classes and functions are included here.
"""

import os

import MDAnalysis as mda
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *

from chimpss.bridgeport.forcefield import ForceFieldHandler
from chimpss.bridgeport.ligand import Ligand
from chimpss.bridgeport.openmm_joiner import Joiner


class IO_Grid:
    """Read and write alchemical grids in dx or netcdf format."""

    def __init__(self):
        pass

    def read(self, FN, multiplier=None):
        if FN is None:
            raise Exception('File is not defined')
        elif FN.endswith('.dx') or FN.endswith('.dx.gz'):
            data = self._read_dx(FN)
        elif FN.endswith('.nc'):
            data = self._read_nc(FN)
        else:
            raise Exception('File type not supported')
        if multiplier is not None:
            data['origin']  = multiplier * data['origin']
            data['spacing'] = multiplier * data['spacing']
        return data

    def _read_dx(self, FN):
        if FN.endswith('.dx'):
            F = open(FN, 'r')
        else:
            import gzip
            F = gzip.open(FN, 'r')

        line = F.readline()
        while line.find('object') == -1:
            line = F.readline()
        header = {}
        header['counts'] = [int(x) for x in line.split(' ')[-3:]]
        for name in ['origin', 'd0', 'd1', 'd2']:
            header[name] = [float(x) for x in F.readline().split(' ')[-3:]]
        F.readline()
        header['npts'] = int(F.readline().split(' ')[-3])

        if not (header['d0'][1] == 0 and header['d0'][2] == 0
                and header['d1'][0] == 0 and header['d1'][2] == 0
                and header['d2'][0] == 0 and header['d2'][1] == 0):
            raise Exception('Trilinear grid must be in original basis')
        if not (header['d0'][0] > 0 and header['d1'][1] > 0
                and header['d2'][2] > 0):
            raise Exception('Trilinear grid must have positive coordinates')

        vals  = np.ndarray(shape=header['npts'], dtype=float)
        index = 0
        while index < header['npts']:
            line  = F.readline()[:-1]
            items = [float(item) for item in line.split()]
            vals[index:index + len(items)] = items
            index += len(items)
        F.close()

        return {
            'origin':  np.array(header['origin']),
            'spacing': np.array([header['d0'][0], header['d1'][1], header['d2'][2]]),
            'counts':  np.array(header['counts']),
            'vals':    vals,
        }

    def _read_nc(self, FN):
        from netCDF4 import Dataset
        grid_nc = Dataset(FN, 'r')
        data = {key: np.array(grid_nc.variables[key][:][0][:])
                for key in list(grid_nc.variables)}
        grid_nc.close()
        return data

    def write(self, FN, data, multiplier=None):
        if multiplier is not None:
            data = {
                'origin':  multiplier * data['origin'],
                'counts':  data['counts'],
                'spacing': multiplier * data['spacing'],
                'vals':    data['vals'],
            }
        if FN.endswith('.nc'):
            self._write_nc(FN, data)
        elif FN.endswith('.dx') or FN.endswith('.dx.gz'):
            self._write_dx(FN, data)
        else:
            raise Exception('File type not supported')

    def _write_dx(self, FN, data):
        n_points = data['counts'][0] * data['counts'][1] * data['counts'][2]
        if FN.endswith('.dx'):
            F = open(FN, 'w')
        else:
            import gzip
            F = gzip.open(FN, 'w')

        F.write("""object 1 class gridpositions counts {0[0]} {0[1]} {0[2]}
origin {1[0]} {1[1]} {1[2]}
delta {2[0]} 0.0 0.0
delta 0.0 {2[1]} 0.0
delta 0.0 0.0 {2[2]}
object 2 class gridconnections counts {0[0]} {0[1]} {0[2]}
object 3 class array type double rank 0 items {3} data follows
""".format(data['counts'], data['origin'], data['spacing'], n_points))

        for start_n in range(0, len(data['vals']), 3):
            F.write(' '.join(['%6e' % c
                              for c in data['vals'][start_n:start_n + 3]]) + '\n')
        F.write('object 4 class field\n')
        F.write('component "positions" value 1\n')
        F.write('component "connections" value 2\n')
        F.write('component "data" value 3\n')
        F.close()

    def _write_nc(self, FN, data):
        n_points = data['counts'][0] * data['counts'][1] * data['counts'][2]
        from netCDF4 import Dataset
        grid_nc = Dataset(FN, 'w', format='NETCDF4')
        grid_nc.createDimension('one', 1)
        grid_nc.createDimension('n_cartesian', 3)
        grid_nc.createDimension('n_points', n_points)
        grid_nc.createVariable('origin',  'f8', ('one', 'n_cartesian'))
        grid_nc.createVariable('counts',  'i8', ('one', 'n_cartesian'))
        grid_nc.createVariable('spacing', 'f8', ('one', 'n_cartesian'))
        grid_nc.createVariable('vals',    'f8', ('one', 'n_points'), zlib=True)
        for key in data:
            grid_nc.variables[key][:] = data[key]
        grid_nc.close()

    def truncate(self, in_FN, out_FN, counts, multiplier=None,
                 in_xyz=None, out_xyz=None):
        data_o = self.read(in_FN)

        if in_xyz is None and out_xyz is None:
            nyz_o = data_o['counts'][1] * data_o['counts'][2]
            nz_o  = data_o['counts'][2]
            min_i = int(-data_o['origin'][0] / data_o['spacing'][0])
            min_j = int(-data_o['origin'][1] / data_o['spacing'][1])
            min_k = int(-data_o['origin'][2] / data_o['spacing'][2])
            vals  = np.array([[[
                data_o['vals'][(i + min_i) * nyz_o + (j + min_j) * nz_o + (k + min_k)]
                for k in range(counts[2])]
                for j in range(counts[1])]
                for i in range(counts[0])])
        else:
            matching_inds = np.empty(out_xyz.shape[0], dtype=int)
            for i, (x, y, z) in enumerate(out_xyz):
                same_x = np.where(in_xyz[:, 0] == x)[0]
                same_y = np.where(in_xyz[:, 1] == y)[0]
                same_z = np.where(in_xyz[:, 2] == z)[0]
                ind = np.intersect1d(np.intersect1d(same_x, same_y), same_z)
                assert len(ind) == 1
                matching_inds[i] = ind[0]
            vals = data_o['vals'][matching_inds]

        if multiplier is not None:
            vals = vals * multiplier

        if out_xyz is not None:
            data_n = {'origin': out_xyz[0], 'counts': counts,
                      'spacing': data_o['spacing'], 'vals': vals.flatten()}
        else:
            data_n = {'origin': np.array([0, 0, 0]), 'counts': counts,
                      'spacing': data_o['spacing'], 'vals': vals.flatten()}
        self.write(out_FN, data_n)


def select_whole_residues(universe, ref_selection):
    """Retrieve whole residues for a selection; returns (included, excluded) groups."""
    group1 = universe.select_atoms(ref_selection)
    group2 = universe.select_atoms('same residue as group group1', group1=group1)
    group3 = universe.select_atoms('not same residue as group group2', group2=group2)
    return group2, group3


def parameterize_from_pdb(pdb_file, lig_smiles, nonbondedMethod=CutoffNonPeriodic,
                          nonbondedCutoff=1.0 * nanometer,
                          lig_resname='UNK', cleanup=True, build_w_obc2=False):
    uni     = mda.Universe(pdb_file)
    not_lig = uni.select_atoms(f'not resname {lig_resname}')
    lig     = uni.select_atoms('resname UNK')

    if not os.path.isdir('./temp/'):
        os.mkdir('./temp/')

    not_lig.write('./temp/dummy_P.pdb')
    lig.write('./temp/dummy_L.pdb')

    ff_xmls = ['amber14/protein.ff14SB.xml',
               'amber14/lipid17.xml',
               'ForceFields/wat_opc3.xml']
    ff = ForceField(*ff_xmls)
    pdb = PDBFile('./temp/dummy_P.pdb')
    rec_top, rec_pos = pdb.getTopology(), pdb.getPositions()
    rec_sys = ff.createSystem(rec_top, nonbondedMethod=nonbondedMethod,
                              nonbondedCutoff=nonbondedCutoff)

    ligand = Ligand(working_dir='./temp/', name='dummy_L', resname='UNK',
                    smiles=lig_smiles)
    ligand.prepare_ligand()
    lig_sys, lig_top, lig_pos = ForceFieldHandler('./temp/dummy_L.sdf').main(
        use_pme='NoCutoff')

    sys, top, pos = Joiner((lig_sys, lig_top, lig_pos),
                            (rec_sys, rec_top, rec_pos)).main()

    if cleanup:
        for fn in ('./temp/dummy_P.pdb', './temp/dummy_L.pdb', './temp/dummy_L.sdf'):
            if os.path.exists(fn):
                os.remove(fn)
        os.rmdir('./temp')

    return sys, top, pos


def simulate_from_filepair(pdb_top_fn, sys_xml_fn,
                           temp=300, ts=0.001, n_steps=int(1e6),
                           dcd_fn='output.dcd', dcd_freq=1000,
                           stdout_fn='output.stdout', stdout_freq=1000,
                           working_dir='./'):
    os.makedirs(working_dir, exist_ok=True)
    pdb = PDBFile(pdb_top_fn)
    with open(sys_xml_fn, 'r') as f:
        system = XmlSerializer.deserialize(f.read())
    integrator = LangevinMiddleIntegrator(temp * kelvin, 1 / picosecond,
                                          ts * picoseconds)
    sim = Simulation(pdb.topology, system, integrator)
    sim.context.setPositions(pdb.positions)
    sim.minimizeEnergy()
    with open(os.path.join(working_dir, 'minimized.pdb'), 'w') as f:
        PDBFile.writeFile(sim.topology,
                          sim.context.getState(getPositions=True).getPositions(),
                          file=f, keepIds=True)
    sim.reporters.append(DCDReporter(os.path.join(working_dir, dcd_fn), dcd_freq))
    sim.reporters.append(StateDataReporter(
        os.path.join(working_dir, stdout_fn), stdout_freq,
        step=True, potentialEnergy=True, temperature=True, speed=True))
    sim.step(n_steps)
    state    = sim.context.getState(getPositions=True, getVelocities=True,
                                    enforcePeriodicBox=True)
    contents = XmlSerializer.serialize(state)
    with open(os.path.join(working_dir, 'final_state.xml'), 'w') as f:
        f.write(contents)
    with open(os.path.join(working_dir, 'final_top.pdb'), 'w') as f:
        PDBFile.writeFile(sim.topology,
                          sim.context.getState(getPositions=True).getPositions(),
                          file=f, keepIds=True)
