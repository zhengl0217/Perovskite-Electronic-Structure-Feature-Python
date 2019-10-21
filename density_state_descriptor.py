import numpy as np

def smooth(y, box_pts):
    """Smooth the noisy density of state distribution using convolution operater.
    Args:
       y (array): one-dimensional input array
       box_pts (int): average box size parameter
    Returns:
       array: one-dimentional array of smoothed density of state values
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def moment(x, y, n):
    """moments function to calculate the porbability distribution characteristics of density of states.
    Args:
       x (array): energy values
       y (array): density of states
       n (int): order parameter of moments function
    Returns:
       float: moment descriptor 
    """
    p = x**n * y
    return np.trapz(p, x)/np.trapz(y, x)

def density_moments(energies, dos):
    """Calculate the moment descriptors for the density of states distributions.
    Args:
       energies (array): energy values with respect to fermi energy
       dos (array): density of states
    returns:
       list: moment characteristics including filling, center, sigma, skewness, kurtosis
    """
    # smooth the noisy density state 
    dos_rev = smooth(dos[:], 15)# smoothing function
    # determine the index of first non-positive value  
    Ind = np.argmax(energies>0)
    # calculate the moment descriptors for the density of states 
    filling = np.trapz(dos_rev[0:Ind], energies[0:Ind])/np.trapz(dos_rev, energies)
    center = moment(energies[:], dos_rev[:], 1)             
    sigma_c = np.sqrt(moment(energies[:]-center, dos_rev[:], 2))
    skewness = moment(energies[:]-center, dos_rev[:], 3)/sigma_c**3 
    kurtosis = moment(energies[:]-center, dos_rev[:], 4)/sigma_c**4 
    return [filling, center, sigma_c, skewness, kurtosis]

def eg_ratio(energies, dos_eg, dos_total):
    """Calculate the ratio of eg orbital filling with respect to the total density of states.
    Args:                                                                                                                        
       energies (array): energy values with respect to fermi energy                                                                       
       dos_eg (array): eg orbital density of states                                                                                     
       dos_total (array): total orbital density of states
    returns:                                                                                                                             
       float: retio of eg orbital electron occupency
    """
    # smooth the noisy density state 
    dos_eg_rev = smooth(dos_eg[:], 15) 
    dos_total_rev = smooth(dos_total[:], 15)
    # determine the index of first non-positive value
    Ind = np.argmax(energies>0)           
    occupancy_eg = np.trapz(dos_eg_rev[0:Ind], energies[0:Ind])
    occupancy_total = np.trapz(dos_total_rev[0:Ind], energies[0:Ind])
    # calculate the ratio of eg orbital electron occupency 
    eg_ratio = occupancy_eg/occupancy_total
    return eg_ratio

