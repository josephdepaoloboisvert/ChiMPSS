#WORK IN PROGRESS
#WORK IN PROGRESS
#WORK IN PROGRESS
#WORK IN PROGRESS
#WORK IN PROGRESS
#WORK IN PROGRESS
#WORK IN PROGRESS
#WORK IN PROGRESS
#WORK IN PROGRESS

# USAGE python Test_Things_Work.py
# Sequentially test each type of Fulton Market Simulation
# Skipping one does not take away a chance to test the next

#Test all imports for custom packages
from FultonMarket.FultonMarketUtils import *
from FultonMarket.Randolph import Randolph
from FultonMarket.FultonMarket import FultonMarket

#Other imports
import os, sys, glob
import openmm.unit as unit

def delete_all_files_in_dir(the_dir):
    files_wildcard = os.path.join(the_dir, '*')
    for f in glob.glob(files_wildcard):
        os.system(f'rm -r {f}')


#Fulton Market Test
response = input('Proceed with testing FultonMarket? y/n \n')
if response == 'y':
    #Setup Block
    test_output_dir = './Test_Cases/FM_test/'
    if not os.path.isdir(test_output_dir):
        os.mkdir(test_output_dir)
    else:
        response = input(f'Should delete the contents of {test_output_dir}? y/n \n')
        if response == 'y':
            delete_all_files_in_dir(test_output_dir)
    
    
    init_kwargs = dict(input_pdb='./Test_Cases/input/7OH.pdb',
                       input_system='./Test_Cases/input/7OH_sys.xml',
                       input_state='./Test_Cases/input/7OH_state.xml',
                       n_replicates=5,
                       T_min=310,
                       T_max=320)
    run_kwargs = dict(convergence_thresh=0.3, resids=np.load('./Test_Cases/input/MOR_resids.npy'), iter_length=0.001, sim_length=0.1,
                      output_dir=test_output_dir, init_overlap_thresh=0.0, term_overlap_thresh=0.1)
    market = FultonMarket(**init_kwargs)
    market.run(**run_kwargs)

