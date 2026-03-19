import numpy as np

def check_boundary_leakage(rho_2d, r_grid, z_grid, threshold=0.01, tol=0.01):
    """
    Validates that mass is not 'leaking' out of the grid boundaries.
    
    Args:
        rho_2d: shape (n_r, n_z), the density field
        r_grid: 1D array of r coordinates (n_r,)
        z_grid: 1D array of z coordinates (n_z,)
        threshold: minimum rho value to count as 'mass' (relative to max rho)
        tol: maximum allowed fraction of total mass at any boundary
    
    Returns:
        dict: leakage stats for each boundary
    """
    n_r, n_z = rho_2d.shape
    
    # Calculate total mass (weighted by r for axisymmetric)
    # rho * r * dr * dz
    dr = r_grid[1] - r_grid[0] if len(r_grid) > 1 else 1.0
    dz = z_grid[1] - z_grid[0] if len(z_grid) > 1 else 1.0
    
    r_2d = r_grid[:, None]
    mass_elements = 2 * np.pi * rho_2d * r_2d * dr * dz
    total_mass = np.sum(mass_elements)
    
    if total_mass == 0:
        return {"total_mass": 0, "leaking": False}

    # Identify boundaries
    # Indices:
    # r=0: [0, :]
    # r=R_ext: [-1, :]
    # z=0: [:, 0]
    # z=z_max: [:, -1]
    
    boundaries = {
        "r_min (0)": mass_elements[0, :],
        "r_max (R_ext)": mass_elements[-1, :],
        "z_min (0)": mass_elements[:, 0],
        "z_max (z_max)": mass_elements[:, -1]
    }
    
    results = {"total_mass": total_mass, "boundary_fractions": {}, "leaking": False}
    
    for name, mass_slice in boundaries.items():
        b_mass = np.sum(mass_slice)
        fraction = b_mass / total_mass
        results["boundary_fractions"][name] = fraction
        
        # We only care about leakage at the 'outer' boundaries usually:
        # r_max and z_max. r_min=0 and z_min=0 are physical boundaries.
        if name in ["r_max (R_ext)", "z_max (z_max)"]:
            if fraction > tol:
                results["leaking"] = True
                
    return results

if __name__ == "__main__":
    # Example usage / test
    nr, nz = 100, 100
    r = np.linspace(0, 3.5, nr)
    z = np.linspace(0, 1.0, nz)
    rho = np.zeros((nr, nz))
    rho[50, 50] = 1.0 # center mass
    
    print("Test 1 (No leakage):")
    print(check_boundary_leakage(rho, r, z))
    
    print("\nTest 2 (Leakage at r_max):")
    rho[-1, 50] = 100.0
    print(check_boundary_leakage(rho, r, z))
