"""
Physical parameters for the Agilicious platform. Values parameterize the inertia, motor dynamics, 
rotor aerodynamics, parasitic drag, and rotor placement. 
Additional sources:
    https://github.com/uzh-rpg/agilicious
    https://www.science.org/doi/abs/10.1126/scirobotics.abl6259
"""
import numpy as np

# Calculate arm length from rotor positions
d = np.sqrt(0.075**2 + 0.10**2)  # Calculated from positions

quad_params = {
    # Inertial properties
    'mass': 0.752,       # kg
    'Ixx':  2.5e-3,     # kg*m^2
    'Iyy':  2.1e-3,     # kg*m^2
    'Izz':  4.3e-3,     # kg*m^2
    'Ixy':  0.0,        # kg*m^2
    'Iyz':  0.0,        # kg*m^2 
    'Ixz':  0.0,        # kg*m^2

    # Geometric properties, all vectors are relative to the center of mass.
    'num_rotors': 4,                        # for looping over each actuator
    'rotor_radius': 0.10,                   # rotor radius, in meters for visualization
    'rotor_pos': {  
                    'r1': np.array([ 0.075,   0.10, 0]),    # Location of Rotor 2 (front left), meters
                    'r2': np.array([ 0.075,  -0.10, 0]),    # Location of Rotor 1 (front right), meters
                    'r3': np.array([-0.075,  -0.10, 0]),    # Location of Rotor 4 (back right), meters
                    'r4': np.array([-0.075,   0.10, 0]),    # Location of Rotor 3 (back left), meters
                 },

    'rotor_directions': np.array([1,-1,1,-1]),  # This dictates the direction of the torque for each motor. 

    'rI': np.array([0,0,0]), # location of the IMU sensor, meters

    # Frame aerodynamic properties - set to 0 as per yaml
    'c_Dx': 0.5e-2,  # parasitic drag in body x axis, N/(m/s)**2
    'c_Dy': 0.5e-2,  # parasitic drag in body y axis, N/(m/s)**2
    'c_Dz': 0.01,  # parasitic drag in body z axis, N/(m/s)**2

    # Rotor properties
    'k_eta': 1.562522e-6,      # thrust coefficient N/(rad/s)**2 from thrust_map[0]
    'k_m': 1.562522e-6 * 0.022,  # yaw moment coefficient Nm/(rad/s)**2 (k_eta * kappa)
    'k_d': 1.19e-04,          # rotor drag coefficient N/(rad*m/s**2) = kg/rad
    'k_z': 2.32e-04,          # induced inflow coefficient N/(rad*m/s**2) = kg/rad
    'k_flap': 0.0,            # Flapping moment coefficient Nm/(rad*m/s**2) = kg*m/rad

    # Motor properties
    'tau_m': 0.033,             # motor response time, seconds
    'rotor_speed_min': 150.0,   # rad/s
    'rotor_speed_max': 2800.0,  # rad/s
    'motor_noise_std': 50,      # rad/s
}
