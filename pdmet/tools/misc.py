#!/usr/bin/env python -u 
'''
pDMET: Density Matrix Embedding theory for Periodic Systems
Copyright (C) 2018 Hung Q. Pham. All Rights Reserved.
A few functions in pDMET are modifed from QC-DMET Copyright (C) 2015 Sebastian Wouters

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Email: Hung Q. Pham <pqh3.14@gmail.com>
'''

import os, datetime
import numpy as np
                   
            
            
BOHR = 0.52917721092
def make_imp_orbs(cell, w90, impCluster, threshold=1.0, rm_list=None, add_list=None):
    '''Attribute:
            cell            : PySCF cell object
            w90             : the w90 object for MLWFs
            impCluster      : a list of the atom labels starting from 1
       Return:
            impOrbs         : list of the MWLFs that belong to the impCluster
       
       ** Note: Unlike the original pDMET code, MLWFs are assigned to atoms beforehand
                regardless of the chosen impurity atoms
       ** Not implemented: rm_list & add_list
    '''
    impCluster = np.asarray(impCluster)

    def put_atoms_in_unitcell(frac_coors):
        coors = frac_coors.flatten()
        coors[coors < 0.0] = coors[coors < 0.0] + 1.0
        coors[coors > 1.0] = coors[coors > 1.0] - 1.0
        return coors.reshape(-1,3)

    assert impCluster.max() <= cell.natm, \
            "Check the impCluster. There are {0} atoms in the unit cell".format(cell.natm)

    # Make sure all the atoms inside the unit cell
    lattice = cell.lattice_vectors() * BOHR
    inv_lattice = np.linalg.inv(lattice)
    abs_coors = cell.atom_coords() * BOHR
    frac_coors = abs_coors @ inv_lattice
    abs_coors = put_atoms_in_unitcell(frac_coors) @ lattice

    # Make sure all the MLWFs inside the unit cell
    num_wann = w90.wann_centres.shape[0]
    MLWFs_coors = w90.wann_centres
    MLWFs_frac_coors = MLWFs_coors @ inv_lattice
    MLWFs_coors = put_atoms_in_unitcell(MLWFs_frac_coors) @ lattice

    # Modified code starts here
    # Calculate the distance between MLWFs and the all atoms
    tmp = np.repeat(MLWFs_coors[:,np.newaxis,:], abs_coors.shape[0], axis=1)
    distance = np.sqrt(np.sum((tmp - abs_coors)**2, axis=2))
    min_distance = distance.min(axis=1)
    min_distance_idx = np.argmin(distance, axis=1)

    impOrbs = np.zeros(num_wann, dtype=int)

    for atom in impCluster:
        impOrbs += np.logical_and(min_distance_idx == (atom - 1), min_distance < threshold)

    # Group the impurity orbitals by their corresponding atoms
    Norbs = MLWFs_coors.shape[0]
    impAtms = []
    for i in impCluster:
        impAtms.append(np.arange(Norbs)[np.logical_and(min_distance_idx == (i-1), min_distance < threshold)])

    return impOrbs, impAtms
