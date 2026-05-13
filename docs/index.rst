ChiMPSS — Chicago Membrane Protein Simulation Suite
====================================================

**ChiMPSS** is a three-stage Python pipeline for preparing and running
molecular-dynamics simulations of membrane proteins (GPCRs and related targets).

.. code-block:: text

   PDB + FASTA + SMILES
          │
          ▼
   ┌─────────────┐
   │  Bridgeport │  system construction → solvated OpenMM XML
   └─────────────┘
          │
          ▼
   ┌─────────────┐
   │  MotorRow   │  5-step NVT/NPT equilibration
   └─────────────┘
          │
          ▼
   ┌──────────────┐
   │ FultonMarket │  parallel-tempering REMD (openmmtools)
   └──────────────┘

Stages are file-coupled only — each stage consumes the previous one's PDB/XML
outputs.  Install the package with ``pip install -e .`` and import each stage
class directly:

.. code-block:: python

   from chimpss.bridgeport import Bridgeport
   from chimpss.motorrow import MotorRow
   from chimpss.fultonmarket import FultonMarket

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api/index
