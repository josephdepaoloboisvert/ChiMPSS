from FultonMarket.analysis import FultonMarketAnalysis

test = FultonMarketAnalysis(input_dir='/expanse/lustre/projects/iit122/josephdb/KOR_Luis/output/RS1125_0/',
                            pdb='/expanse/lustre/projects/iit122/josephdb/KOR_Luis/Luis_Data/RS1125.pdb',
                            sele_str='resname UNK')

import os
matrices = test.retro_analyze_all(n_resample=1000,
                                  output_cache_dir='/expanse/lustre/projects/uil133/josephdb/Analysis_Data/KOR_RS1125_retro/rpt_0/',
                                  getcontacts_script='/expanse/lustre/projects/uil133/josephdb/getcontacts/get_dynamic_contacts.py',
                                  conda_env='pyinteraph2',
                                  getcontacts_python=os.path.join(os.environ['CONDA_PREFIX'].replace('replica2','pyinteraph2'),'bin','python'))

test.retro_convergence_report(total_sim_time=1000,
                              sim_length=50,
                              read_only=False,
                              n_resample=1000,
                              output_cache_dir='/expanse/lustre/projects/uil133/josephdb/Analysis_Data/KOR_RS1125_retro/rpt_0/')
