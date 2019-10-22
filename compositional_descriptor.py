import numpy as np
import pymatgen as mg

def elemental_descriptor(A1_ion, A2_ion, B_ion):
    """Extract elemental descriptors according to the atomic properties of A1_ion, A2_ion and B_ion in the perovskite structure.
    Args:
       A1_ion (str): element at A1_ion site, e.g., "La"
       A2_ion (str): element at A2_ion site, e.g., "Ba" 
       B_ion (str): element at B_ion site, e.g., "Co" 
    Returns:
       array: compositional descriptors, including common oxidation state of A, common oxidation state of B, Pauling electronegativity of A, Pauling electronegativity B, Tolerance factor, Octahedral factor, ionic ration of A/O, ionic ration of B/O, electronegativity difference of A/O, electronegativity difference of B/O.
    """
    ele_A1 = mg.Element(A1_ion)
    ele_A2 = mg.Element(A2_ion)
    ele_B = mg.Element(B_ion)
    ele_O = mg.Element('O')   
    # A/B ion oxidation state 
    common_oxidation_states_A1 = ele_A1.common_oxidation_states[0]
    common_oxidation_states_A2 = ele_A2.common_oxidation_states[0]
    common_oxidation_states_A = np.mean(common_oxidation_states_A1 + common_oxidation_states_A2)
    common_oxidation_states_B = ele_B.common_oxidation_states[0]
    # ionic radius property
    ionic_radius_A1 = float(str(ele_A1.average_ionic_radius)[:-4])
    ionic_radius_A2 = float(str(ele_A2.average_ionic_radius)[:-4])
    ionic_radius_A = (ionic_radius_A1+ ionic_radius_A2)/2
    ionic_radius_B = float(str(ele_B.average_ionic_radius)[:-4])
    ionic_radius_O = float(str(ele_O.average_ionic_radius)[:-4])
    # Tolerance factor 
    TF = (ionic_radius_A + ionic_radius_O)/(np.sqrt(2)*(ionic_radius_B + ionic_radius_O))
    # Octahedral factor
    OF = ionic_radius_B/ionic_radius_O  
    # ionic_radius ratios
    ionic_ration_AO = ionic_radius_A / ionic_radius_O
    ionic_ration_BO = ionic_radius_B / ionic_radius_O
    # averaged electronegativity for A and B atoms
    Pauling_electronegativity_A1 = ele_A1.X
    Pauling_electronegativity_A2 = ele_A2.X
    Pauling_electronegativity_A = (Pauling_electronegativity_A1 + Pauling_electronegativity_A2)/2
    Pauling_electronegativity_B = ele_B.X
    Pauling_electronegativity_O = ele_O.X
    # Difference in the electronegativity for A-O and B-O
    Diff_A_O = Pauling_electronegativity_A - Pauling_electronegativity_O
    Diff_B_O = Pauling_electronegativity_B - Pauling_electronegativity_O
    return [common_oxidation_states_A, common_oxidation_states_B, Pauling_electronegativity_A, Pauling_electronegativity_B, TF, OF, ionic_ration_AO, ionic_ration_BO, Diff_A_O, Diff_B_O]

def generalized_mean(x, p, N):
    """Generalized mean function capture the mean value of atomic properties by considering the ratio of each element in the structure.                                                                                     
    Args:                                                                                                  
       x (array): array of atomic properties for each atom in the structure.                                           p (int): power parameter, e.g., harmonic mean (-1), geometric mean(0), arithmetic mean(1), quadratic mean(2). 
       N (int): total number of atoms in the structure.
    Returns:                                                                                                 
       float: generalized mean value.                                                                        
    """
    if p != 0:
        D = 1/(N)
        out = (D*sum(x**p))**(1/p)
    else:
        D = 1/(N)
        out = np.exp(D*sum(np.log(x)))
    return out

def geometric_descriptor(element_dict):
    """Extract geometric mean of the atomic properties in the perovskite structure.                       
    Args:                                                                                                    
       element_dict (dict): element frequency library in a perovskite structure, e.g., {'La': 4, 'Ba': 4, 'Co': 8, 'O': 24}                            
    Returns:                                                                                                 
       array: geometric based descriptor, including atomic_radius,  mendeleev number', common_oxidation_states, Pauling electronegativity, thermal_conductivity, average_ionic_radius, atomic_orbitals.               
    """
    # encode the orbital types
    category = {'s': 1, 'p': 2, 'd': 3, 'f': 4};
    # total number of atoms in a perovskite structure
    N = sum(element_dict.values())
    # obtain array of atomic properties for each element type
    atomic_number_list = []
    atomic_mass_list = []
    atomic_radius_list = []
    mendeleev_no_list = []
    common_oxidation_states_list = []
    Pauling_electronegativity_list = []
    row_list = []
    group_list = []
    block_list = []
    thermal_conductivity_list = []
    boiling_point_list = []
    melting_point_list = []
    average_ionic_radius_list = []
    molar_volume_list = []
    atomic_orbitals_list = []
    for item in element_dict:
        # extract atomic property from pymatgen
        ele = mg.Element(item)
        atomic_number = ele.Z
        atomic_mass = float(str(ele.atomic_mass)[:-4])
        atomic_radius = float(str(ele.atomic_radius)[:-4])
        mendeleev_no = ele.mendeleev_no
        common_oxidation_states = ele.common_oxidation_states[0]
        Pauling_electronegativity = ele.X
        row = ele.row
        group = ele.group
        block = ele.block
        thermal_conductivity = float(str(ele.thermal_conductivity)[:-12])
        boiling_point = float(str(ele.boiling_point)[: -2])
        melting_point = float(str(ele.melting_point)[: -2])
        average_ionic_radius = float(str(ele.average_ionic_radius)[:-4])
        molar_volume = float(str(ele.molar_volume)[: -5])
        if '6s' in ele.atomic_orbitals.keys():
            atomic_orbitals = ele.atomic_orbitals['6s']
        elif '4s' in ele.atomic_orbitals.keys():
            atomic_orbitals = ele.atomic_orbitals['4s']
        else:
            atomic_orbitals = ele.atomic_orbitals['2s']
        # calculate the array of atomic properties for all atoms 
        atomic_number_list += [atomic_number]*element_dict[item]
        atomic_mass_list += [atomic_mass]*element_dict[item]
        atomic_radius_list += [atomic_radius]*element_dict[item]
        mendeleev_no_list += [mendeleev_no]*element_dict[item]
        common_oxidation_states_list += [common_oxidation_states]*element_dict[item]
        Pauling_electronegativity_list += [Pauling_electronegativity]*element_dict[item]
        row_list += [row]*element_dict[item]
        group_list += [group]*element_dict[item]
        block_list += [category[block]]*element_dict[item]
        thermal_conductivity_list += [thermal_conductivity]*element_dict[item]
        boiling_point_list += [boiling_point]*element_dict[item]
        melting_point_list += [melting_point]*element_dict[item]
        average_ionic_radius_list += [average_ionic_radius]*element_dict[item]
        molar_volume_list += [molar_volume]*element_dict[item]
        atomic_orbitals_list += [atomic_orbitals]*element_dict[item]
    return [generalized_mean(np.array(atomic_number_list), 1, N)] + [generalized_mean(np.array(atomic_radius_list), 1, N)] + [generalized_mean(np.array(mendeleev_no_list), 1, N)] + [generalized_mean(np.array(common_oxidation_states_list), 1, N)] + [generalized_mean(np.array(Pauling_electronegativity_list), 1, N)] + [generalized_mean(np.array(thermal_conductivity_list), 1, N)] + [generalized_mean(np.array(average_ionic_radius_list), 1, N)] + [generalized_mean(np.array(atomic_orbitals_list), 1, N)]

if __name__ == "__main__":
    print('LaBaCo2O6', elemental_descriptor('La', 'Ba', 'Co') + geometric_descriptor({'La': 4, 'Ba': 4, 'Co': 8, 'O': 24}))
