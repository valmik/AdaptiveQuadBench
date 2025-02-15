# Import your controller here
from controller.geometric_control import GeoControl
from controller.geometric_adaptive_controller import GeometricAdaptiveController
from controller.geometric_control_l1 import L1_GeoControl
from controller.indi_adaptive_controller import INDIAdaptiveController
from controller.quadrotor_control_mpc import ModelPredictiveControl
from controller.quadrotor_control_mpc_l1 import L1_ModelPredictiveControl
from controller.Xadap_NN_control import Xadap_NN_control
# Import your vehicle here
from quad_param.Agilicious import quad_params

from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.world import World
from rotorpy.environments import Environment
from rotorpy.wind.dryden_winds import DrydenGust

from parallel_data_collection import generate_data

import time
import os
import csv
import numpy as np
import pandas as pd

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', type=str, default='geo', help='controller type: geo, geo-a, l1geo, l1mpc, indi-a, xadap')
    parser.add_argument('--experiment', type=str, default='wind', help='experiment type: wind, uncertainty')
    parser.add_argument('--num_trials', type=int, default=100)
    parser.add_argument('--parallel', type=bool, default=True, help='if run in parallel')
    parser.add_argument('--seed', type=int, default=42, help='seed for random number generator')
    parser.add_argument('--save_trials', type=bool, default=False, help='save individual trials to csv')
    return parser.parse_args()

def switch_controller(controller_type,quad_params):
    if controller_type == 'geo':
        return GeoControl(quad_params)
    elif controller_type == 'geo-a':
        return GeometricAdaptiveController(quad_params)
    elif controller_type == 'l1geo':
        return L1_GeoControl(quad_params)
    elif controller_type == 'indi-a':
        return INDIAdaptiveController(quad_params)
    elif controller_type == 'l1mpc':
        return L1_ModelPredictiveControl(quad_params)
    elif controller_type == 'xadap':
        return Xadap_NN_control(quad_params)
    else:
        raise ValueError(f"Controller type {controller_type} not supported yet")

def create_wind_profiles(num_profiles, seed=None):
    """Pre-generate multiple wind profiles"""
    if seed is not None:
        np.random.seed(seed)
    
    wind_profiles = []
    for i in range(num_profiles):
        wind_mean = np.random.uniform(-2, 2, size=3)
        wind_var = np.random.uniform(0.1, 1.0, size=3)
        # Randomize the wind input for this trial
        wx = np.random.uniform(low=-3, high=3)
        wy = np.random.uniform(low=-3, high=3)
        wz = np.random.uniform(low=-3, high=3)
        sx = np.random.uniform(low=30, high=60)
        sy = np.random.uniform(low=30, high=60)
        sz = np.random.uniform(low=30, high=60)
        wind_profile = DrydenGust(dt=1/100, avg_wind=np.array([wx,wy,wz]), sig_wind=np.array([sx,sy,sz]))
        wind_profiles.append(wind_profile)
    
    return wind_profiles

def create_randomized_controllers(controller_type, base_params, num_controllers, seed=None):
    """Pre-generate multiple controller instances with randomized parameter assumptions"""
    if seed is not None:
        np.random.seed(seed)
    
    controllers = []
    for i in range(num_controllers):
        params = base_params.copy()
        # TODO
        # Add random perturbations to controller's parameter assumptions
        # params['mass'] *= (1 + np.random.normal(0, 0.1))  # 10% variation
        params['Ixx'] *= (1 + np.random.normal(0, 0.1))
        params['Iyy'] *= (1 + np.random.normal(0, 0.1))
        params['Izz'] *= (1 + np.random.normal(0, 0.1))
        
        controllers.append(switch_controller(controller_type, params))
    
    return controllers

def generate_summary(controller_type, controllers, vehicle, wind_profiles,
                      num_simulations=100, parallel_bool=True, save_trials=False):
    """
    Main function for generating data.
    Inputs:
        controller: The controller to use.
        vehicle: The vehicle to use.
        wind_profile: The wind profile to use.
        num_simulations: The number of simulations to run.
        parallel_bool: If True, runs the simulations in parallel. If False, runs the simulations sequentially.
        save_trials: If True, saves each trial data to a separate .csv file. Uses more memory, but allows you to see the results of each trial at a later date.
    Outputs:
        None. It writes to the output file.
    """

    world_size = 10
    num_waypoints = 4
    vavg = 2
    random_yaw = False
    yaw_min = -0.85*np.pi
    yaw_max = 0.85*np.pi

    world_buffer = 2
    min_distance = 1
    max_distance = min_distance+3
    start_waypoint = None               # If you want to start at a specific waypoint, specify it using [xstart, ystart, zstart]
    end_waypoint = None                 # If you want to end at a specific waypoint, specify it using [xend, yend, zend]

    # convert controller to string
    controller_name = str(controller_type)

    # Create the output file and handle existing files
    output_csv_file = os.path.dirname(__file__) + f'/data/summary_{controller_name}.csv'

    if os.path.exists(output_csv_file):
        # Ask the user if they want to remove the existing file.
        user_input = input("The file {} already exists. Do you want to remove the existing file? (y/n)".format(output_csv_file))
        if user_input == 'y':
            # Remove the existing file
            os.remove(output_csv_file)
        elif user_input == 'n':
            raise Exception("Please delete or rename the file {} before running this script.".format(output_csv_file))
        else:
            raise Exception("Invalid input. Please enter 'y' or 'n'.")
    
    savepath = None
    if save_trials:
        savepath = os.path.dirname(__file__)+ f'/data/trial_data_{controller_name}'
        print(f"savepath: {savepath}")
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        else:
            # Ask the user if they want to remove the existing files in the directory.
            user_input = input(f"The directory {savepath} already exists. Do you want to remove the existing files? (y/n)")
            if user_input == 'y':
                # Remove existing files in the directory
                for file in os.listdir(savepath):
                    os.remove(os.path.join(savepath, file))
            elif user_input == 'n':
                raise Exception(f"Please delete or rename the files in the directory {savepath} before running this script.")
            else:
                raise Exception("Invalid input. Please enter 'y' or 'n'.")

    # Append headers to the output file
    with open(output_csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # This depends on the number of waypoints and the order of the polynomial. Currently pos is 7th order and yaw is 7th order.
        writer.writerow(['traj_number'] + ['pos_tracking_error'] + ['heading_error'] 
                + ['x_poly_seg_{}_coeff_{}'.format(i,j) for i in range(num_waypoints-1) for j in range(8)]
                + ['y_poly_seg_{}_coeff_{}'.format(i,j) for i in range(num_waypoints-1) for j in range(8)]
                + ['z_poly_seg_{}_coeff_{}'.format(i,j) for i in range(num_waypoints-1) for j in range(8)]
                + ['yaw_poly_seg_{}_coeff_{}'.format(i,j) for i in range(num_waypoints-1) for j in range(8)])  

    # Create world
    world = World.empty([-world_size/2, world_size/2, -world_size/2, world_size/2, -world_size/2, world_size/2])

    # Generate the data
    start_time = time.time()
    generate_data(output_csv_file, world, vehicle, controllers, wind_profiles,
                  num_simulations, num_waypoints, vavg, 
                  random_yaw, yaw_min, yaw_max, 
                  world_buffer, min_distance, max_distance, 
                  start_waypoint, end_waypoint,
                  parallel=parallel_bool,
                  save_individual_trials=save_trials,
                  save_trial_path=savepath)
    end_time = time.time()
    print("Time elapsed: %3.2f seconds, parallel: %s" % (end_time-start_time, parallel_bool))

    # Print final summary
    df = pd.read_csv(output_csv_file)
    
    # count df['pos_tracking_error'] < 5
    success_rate = (df['pos_tracking_error'] < 5).sum() / len(df) * 100 
    # successful filter
    pos_tracking_error_success = df['pos_tracking_error'] < 5
    
    print("--------------------------------")
    print(f"Controller: {controller_name}")
    print("--------------------------------")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Average pos_tracking_error: {df['pos_tracking_error'][pos_tracking_error_success].mean():.2f} m")
    print(f"Average heading_error: {df['heading_error'][pos_tracking_error_success].mean():.2f} deg")
    print("--------------------------------")

    return None

def main():
    args = parse_args()
    vehicle = Multirotor(quad_params)
    # TODO Trajectory class of ecllipse
    
    # Create controllers and wind profiles based on experiment type
    if args.experiment == 'wind':
        # Create standard controllers with no parameter variation
        controllers = [switch_controller(args.controller, quad_params) for _ in range(args.num_trials)]
        # Create varied wind profiles
        wind_profiles = create_wind_profiles(args.num_trials, seed=args.seed)
    
    elif args.experiment == 'uncertainty':
        # Create controllers with parameter variations
        controllers = create_randomized_controllers(args.controller, quad_params, args.num_trials, seed=args.seed)
        # Create standard (or no) wind profiles
        wind_profiles = [None] * args.num_trials
    
    else:
        raise ValueError(f"Experiment type {args.experiment} not supported")

    generate_summary(args.controller, controllers, vehicle, wind_profiles, 
                    args.num_trials, args.controller != 'xadap', args.save_trials)

if __name__ == '__main__':
    main()