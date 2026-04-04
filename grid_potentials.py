import MDAnalysis as mda
import mdtraj as md
import os, sys, glob
import numpy as np

from openmm import *
from openmm.app import *
from openmm.unit import *

#sys.path.append('/media/volume/Josephs-Volume/githubs/Bridgeport/')

import ForceFields.force_fields as ff_handler
from Ligand.Ligand import Ligand
from ForceFields.joiner import Joiner

from MotorRow import MotorRow

import multiprocessing as mp
max_cpu=60

class IO_Grid:
  """
  Class to read and write alchemical grids.

  Data is a dictionary with
  spacing - the grid spacing, in Angstroms.
  counts - the number of points in each dimension.
  vals - the values.
  All are numpy arrays.
  """
  def __init__(self):
    pass

  def read(self, FN, multiplier=None):
    """
    Reads a grid in dx or netcdf format
    The multiplier affects the origin and spacing.
    """
    if FN is None:
      raise Exception('File is not defined')
    elif FN.endswith('.dx') or FN.endswith('.dx.gz'):
      data = self._read_dx(FN)
    elif FN.endswith('.nc'):
      data = self._read_nc(FN)
    else:
      raise Exception('File type not supported')
    if multiplier is not None:
      data['origin'] = multiplier * data['origin']
      data['spacing'] = multiplier * data['spacing']
    return data

  def _read_dx(self, FN):
    """
    Reads a grid in dx format
    """
    if FN.endswith('.dx'):
      F = open(FN, 'r')
    else:
      import gzip
      F = gzip.open(FN, 'r')

    # Read the header
    line = F.readline()
    while line.find('object') == -1:
      line = F.readline()
    header = {}
    header['counts'] = [int(x) for x in line.split(' ')[-3:]]
    for name in ['origin', 'd0', 'd1', 'd2']:
      header[name] = [float(x) for x in F.readline().split(' ')[-3:]]
    F.readline()
    header['npts'] = int(F.readline().split(' ')[-3])

    # Test to make sure the grid type is okay.
    # These conditions are not absolultely essential,
    #   but they reduce the number of subtraction operations.
    if not (header['d0'][1] == 0 and header['d0'][2] == 0
            and header['d1'][0] == 0 and header['d1'][2] == 0
            and header['d2'][0] == 0 and header['d2'][1] == 0):
      raise Exception('Trilinear grid must be in original basis')
    if not (header['d0'][0] > 0 and header['d1'][1] > 0
            and header['d2'][2] > 0):
      raise Exception('Trilinear grid must have positive coordinates')

    # Read the data
    vals = np.ndarray(shape=header['npts'], dtype=float)
    index = 0
    while index < header['npts']:
      line = F.readline()[:-1]
      items = [float(item) for item in line.split()]
      vals[index:index + len(items)] = items
      index = index + len(items)
    F.close()

    data = {
      'origin':np.array(header['origin']), \
      'spacing':np.array([header['d0'][0],header['d1'][1],header['d2'][2]]), \
      'counts':np.array(header['counts']), \
      'vals':vals}
    return data

  def _read_nc(self, FN):
    """
    Reads a grid in netcdf format
    """
    from netCDF4 import Dataset
    grid_nc = Dataset(FN, 'r')
    data = {}
    for key in list(grid_nc.variables):
      data[key] = np.array(grid_nc.variables[key][:][0][:])
    grid_nc.close()
    return data

  def write(self, FN, data, multiplier=None):
    """
    Writes a grid in dx or netcdf format.
    The multiplier affects the origin and spacing.

    """
    if multiplier is not None:
      data_n = {
        'origin': multiplier * data['origin'],
        'counts': data['counts'],
        'spacing': multiplier * data['spacing'],
        'vals': data['vals']
      }
    else:
      data_n = data
    if FN.endswith('.nc'):
      self._write_nc(FN, data_n)
    elif FN.endswith('.dx') or FN.endswith('.dx.gz'):
      self._write_dx(FN, data_n)
    else:
      raise Exception('File type not supported')

  def _write_dx(self, FN, data):
    """
    Writes a grid in dx format
    """
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
    """
    Writes a grid in netcdf format
    """
    n_points = data['counts'][0] * data['counts'][1] * data['counts'][2]
    from netCDF4 import Dataset
    grid_nc = Dataset(FN, 'w', format='NETCDF4')
    grid_nc.createDimension('one', 1)
    grid_nc.createDimension('n_cartesian', 3)
    grid_nc.createDimension('n_points', n_points)
    grid_nc.createVariable('origin', 'f8', ('one', 'n_cartesian'))
    grid_nc.createVariable('counts', 'i8', ('one', 'n_cartesian'))
    grid_nc.createVariable('spacing', 'f8', ('one', 'n_cartesian'))
    grid_nc.createVariable('vals', 'f8', ('one', 'n_points'), zlib=True)
    for key in data.keys():
      grid_nc.variables[key][:] = data[key]
    grid_nc.close()

  def truncate(self, in_FN, out_FN, counts, multiplier=None, in_xyz=None, out_xyz=None):
    """
    Truncates the grid at the origin and
    with a limited number of counts per dimension

    multiplier is for the values, not the grid scaling
    """
    data_o = self.read(in_FN)

    if (in_xyz is None and out_xyz is None):
        nyz_o = data_o['counts'][1] * data_o['counts'][2]
        nz_o = data_o['counts'][2]
    
        print('origin', data_o['origin'])
        print('spacing', data_o['spacing'])
    
        min_i = int(-data_o['origin'][0] / data_o['spacing'][0])
        min_j = int(-data_o['origin'][1] / data_o['spacing'][1])
        min_k = int(-data_o['origin'][2] / data_o['spacing'][2])
    
        print('min_i', min_i)
        print('min_j', min_j)
        print('min_k', min_k)
        print('vals.shape', data_o['vals'].shape)
    
        # vals = np.ndarray(shape=tuple(counts), dtype=float)
        # print 'vals.shape', vals.shape 
        # for i in range(counts[0]):
        #   for j in range(counts[1]):
        #     for k in range(counts[2]):
        #       # print 'i,j,k', i, j, k, (i+min_i)*nyz_o + (j+min_j)*nz_o + (k+min_k)
        #       vals[i,j,k] = data_o['vals'][(i+min_i)*nyz_o + (j+min_j)*nz_o + (k+min_k)]

        vals = np.array([[[data_o['vals'][(i + min_i) * nyz_o + (j + min_j) * nz_o + (k + min_k)] for k in range(counts[2]) ] for j in range(counts[1])] for i in range(counts[0])])

    else:
        matching_inds = np.empty(out_xyz.shape[0], dtype=int)
        for i, (x, y, z) in enumerate(out_xyz):
            same_x_inds = np.where(in_xyz[:,0] == x)[0]
            same_y_inds = np.where(in_xyz[:,1] == y)[0]
            same_z_inds = np.where(in_xyz[:,2] == z)[0]
            xy_ind = np.intersect1d(same_x_inds, same_y_inds)
            ind = np.intersect1d(same_z_inds, xy_ind)
            assert len(ind) == 1
            matching_inds[i] = ind[0]

        vals = data_o['vals'][matching_inds]
        

    if multiplier is not None:
      vals = vals * multiplier

    if out_xyz is not None:
        data_n = {'origin': out_xyz[0], \
          'counts':counts, 'spacing':data_o['spacing'], 'vals':vals.flatten()}
    else:
        data_n = {'origin': np.array([0,0,0]), \
          'counts':counts, 'spacing':data_o['spacing'], 'vals':vals.flatten()}
    self.write(out_FN, data_n)


def select_whole_residues(universe, ref_selection):
        """
        universe = mdanalysis universe to select from
        atom_selection = mdanalysis selection string for included atoms

        returns group_included, group_excluded
        
        Retrieve whole residues for a selection
        """
        group1 = universe.select_atoms(ref_selection)
        group2 = universe.select_atoms('same residue as group group1', group1=group1)
        group3 = universe.select_atoms('not same residue as group group2', group2=group2)
        return group2, group3

def parameterize_from_pdb(pdb_file, lig_smiles, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1.0*nanometer,
                          lig_resname='UNK', cleanup=True, build_w_obc2=False):

    #Seperate the stuff
    uni = mda.Universe(pdb_file)
    print(len(uni.atoms))
    not_lig = uni.select_atoms(f'not resname {lig_resname}')
    lig = uni.select_atoms('resname UNK')

    if not os.path.isdir('./temp/'):
        os.mkdir('./temp/')
    
    not_lig.write('./temp/dummy_P.pdb')
    lig.write('./temp/dummy_L.pdb')

    #Get the Parameters
    ff_xmls = ['amber14/protein.ff14SB.xml',
               'amber14/lipid17.xml',
               'ForceFields/wat_opc3.xml']

    #if build_w_obc2:
    #    ff_xmls.append('implicit/obc2.xml')
    
    ff = ForceField(*ff_xmls)
    pdb = PDBFile('./temp/dummy_P.pdb')
    rec_top, rec_pos = pdb.getTopology(), pdb.getPositions()
    rec_sys = ff.createSystem(rec_top, nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff)
    
    #Ligand Too
    ligand = Ligand(working_dir='./temp/', name='dummy_L', resname='UNK', smiles=lig_smiles)
    ligand.prepare_ligand()
    lig_sys, lig_top, lig_pos = ff_handler.ForceFieldHandler('./temp/dummy_L.sdf').main(use_pme='NoCutoff')
    
    #Joiner
    sys, top, pos = Joiner((lig_sys, lig_top, lig_pos), (rec_sys, rec_top, rec_pos)).main()
    print(sys.getNumParticles())
    for force in sys.getForces():
        if hasattr(force, 'getNonbondedMethod'):
            print(force)
            print(force.getNonbondedMethod(), force.getNumParticles())
    
    if cleanup:
        os.remove('./temp/dummy_P.pdb')
        os.remove('./temp/dummy_L.pdb')
        os.remove('./temp/dummy_L.sdf')
        os.rmdir('./temp')
    
    return sys, top, pos

def simulate_from_filepair(pdb_top_fn, sys_xml_fn,
                           temp=300, ts=0.001, n_steps=1e6, #1ns
                           dcd_fn='output.dcd', dcd_freq=1000,
                           stdout_fn='output.stdout', stdout_freq=1000,
                           working_dir='./'):
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir, exist_ok=True)
    #Construct
    pdb = PDBFile(pdb_top_fn)
    with open(sys_xml_fn, 'r') as f:
        system = XmlSerializer.deserialize(f.read())
    integrator = LangevinMiddleIntegrator(temp*kelvin, 1/picosecond, ts*picoseconds)
    sim = Simulation(pdb.topology, system, integrator)
    _ = sim.context.setPositions(pdb.positions)
    #Minimize
    _ = sim.minimizeEnergy()
    with open(os.path.join(working_dir,'minimized.pdb'), 'w') as f:
        _ = PDBFile.writeFile(sim.topology, sim.context.getState(getPositions=True).getPositions(), file=f, keepIds=True)
    #Simulate
    _ = sim.reporters.append(DCDReporter(os.path.join(working_dir, dcd_fn), dcd_freq))
    _ = sim.reporters.append(StateDataReporter(os.path.join(working_dir, stdout_fn), stdout_freq, step=True, potentialEnergy=True, temperature=True, speed=True))
    _ = sim.step(n_steps)

    #Final
    state = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    contents = XmlSerializer.serialize(state)
    with open(os.path.join(working_dir, 'final_state.xml'), 'w') as f:
        _ = f.write(contents)
    with open(os.path.join(working_dir,'final_top.pdb'), 'w') as f:
        _ = PDBFile.writeFile(sim.topology, sim.context.getState(getPositions=True).getPositions(), file=f, keepIds=True)
    return None

from datetime import datetime
#import jax
#import jax.numpy as jnp

pdb_fn = 'Static_Env/MotorRowBenchMark/Step_5_wrapped.pdb'
#ligand_smiles = "CCN(CC)C(=O)N[C@@H]1CN([C@@H]2Cc3c[nH]c4c3c(ccc4)C2=C1)C"
ligand_smiles = "[H][C@]1(C[C@@H]2[C@H]3C(=C[C@]4(CCCC[C@@]24[H])O[C@@]13C)C1=NCCO1)C1=CC=CC(C)=N1"
pdb = PDBFile(pdb_fn)
u = mda.Universe(pdb_fn)

envelope, outside = select_whole_residues(u, 'protein or resname UNK or (around 5 (protein or resname UNK))')
envelope, outside, envelope.indices.shape, outside.indices.shape

ref_sys, top, pos = parameterize_from_pdb('Static_Env/MotorRowBenchMark/Step_5_wrapped.pdb', lig_smiles=ligand_smiles)

print(envelope.positions.max(axis=0), envelope.positions.min(axis=0))
print(outside.positions.max(axis=0), outside.positions.min(axis=0))

indices, adj_pos = np.array(outside.indices), np.array(outside.positions) - outside.positions.min(axis=0)

dim = adj_pos.max(axis=0) - adj_pos.min(axis=0) #size
spacing = 0.125 #Angstrom desired spacing
#dim *= 0.1 #Angstrom to nanometer if necessary
counts = np.ceil(dim/spacing).astype(int)
spacing = [spacing]*3

adj_pos.min(axis=0)

### Coordinates of grid points
print('Calculating grid coordinates')
startTime = datetime.now()

# xyz = get_grid_pos(center, counts, spacing)
grid = {}
grid['x'] = np.zeros(shape=tuple(counts), dtype=float)
grid['y'] = np.zeros(shape=tuple(counts), dtype=float)
grid['z'] = np.zeros(shape=tuple(counts), dtype=float)
for i in range(counts[0]):
    if i % (counts[0]//10) == 0:
        print(i)
    
    for j in range(counts[1]):
        for k in range(counts[2]):
            grid['x'][i,j,k] = i*spacing[0]
            grid['y'][i,j,k] = j*spacing[1]
            grid['z'][i,j,k] = k*spacing[2]

# for key in ['x', 'y', 'z']:
#    grid[key] = jnp.array(grid[key])

endTime = datetime.now()
print(f' in {endTime-startTime}')

#These points are however relative to the origin...

n_points = np.prod(counts)
n_atoms = indices.shape[0]

nbf = [force for force in ref_sys.getForces() if type(force) == NonbondedForce][0]
startTime = datetime.now()
print("Reading Parameters for Grid Particles")
charges = np.zeros(n_atoms) #elementary charge
depths = np.zeros(n_atoms) #OpenMM = kJ/mol convert to AMBER kcal/mol
radii = np.zeros(n_atoms) #OpenMM = nm convert to AMBER Angstrom
for i, atom_index in enumerate(indices):
    q, r, e = nbf.getParticleParameters(atom_index)
    charges[i] = q.value_in_unit(elementary_charge)
    depths[i] = e.value_in_unit(kilocalorie_per_mole)
    radii[i] = r.value_in_unit(angstrom)

root_depths = np.sqrt(depths)
diameters = 2*radii

endTime = datetime.now()
print(f' in {endTime-startTime}')

points = np.array([grid['x'].flatten(), grid['y'].flatten(), grid['z'].flatten()]).T

print('Calculating Grid Potentials')
startTime = datetime.now()

coulomb_flat = np.zeros(points.shape[0])
ljr_flat = np.zeros(points.shape[0])
lja_flat = np.zeros(points.shape[0])

def distance_all_to_point(p):
    return np.sqrt(np.sum((adj_pos - p)**2, axis=-1))

def assign_grid_point(i):
    p = points[i]
    dist = distance_all_to_point(p)
    coulomb_flat[i] = np.sum(332.06*charges/dist)
    ljr_flat[i] = np.sum(root_depths*(diameters**6)/(dist**12))
    lja_flat[i] = np.sum(-2*root_depths*(diameters**3)/(dist**6))

with mp.Pool(processes=mp.cpu_count()) as pool:
    _ = pool.map(assign_grid_point, np.arange(points.shape[0]), chunksize=10000)

endTime = datetime.now()
print(f' in {endTime-startTime}')

np.save('test_coulomb.npy', coulomb_flat)
np.save('test_ljr.npy', ljr_flat)
np.save('test_lja.npy', lja_flat)
