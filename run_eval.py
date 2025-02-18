# Import your controller here
from controller.geometric_control import GeoControl
from controller.geometric_adaptive_controller import GeometricAdaptiveController
from controller.geometric_control_l1 import L1_GeoControl
from controller.indi_adaptive_controller import INDIAdaptiveController
from controller.quadrotor_control_mpc import ModelPredictiveControl
from controller.quadrotor_control_mpc_l1 import L1_ModelPredictiveControl
from controller.Xadap_NN_control import Xadap_NN_control
from rotorpy.controllers.quadrotor_control import SE3Control
# Import your vehicle here
from quad_param.Agilicious import quad_params
from rotorpy.trajectories.random_motion_prim_traj import RapidTrajectory

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
from pathlib import Path
from randomization_config import RandomizationConfig, ExperimentType

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', type=str, default='geo', 
                       help='controller type: geo, geo-a, l1geo, l1mpc, indi-a, xadap')
    parser.add_argument('--experiment', type=str, default='wind', 
                       choices=[e.value for e in ExperimentType],
                       help='experiment type: wind, uncertainty, force, torque, all')
    parser.add_argument('--num_trials', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42, 
                       help='seed for random number generator')
    parser.add_argument('--save_trials', type=bool, default=False, 
                       help='save individual trials to csv')
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
    elif controller_type == 'mpc':
        return ModelPredictiveControl(quad_params)
    elif controller_type == 'xadap':
        return Xadap_NN_control(quad_params)
    else:
        print(f"Controller type {controller_type} not supported yet. We use default SE3Control")
        return SE3Control(quad_params)

def generate_summary(controller_type, controllers, vehicle, wind_profiles, trajectories, ext_force, ext_torque,
                      num_simulations=100, parallel_bool=True, save_trials=False):
    """
    Main function for generating data.
    Inputs:
        controller: The controller to use.
        vehicle: The vehicle to use.
        wind_profile: The wind profile to use.
        trajectories: The trajectories to use.
        ext_force: The external force to use.
        ext_torque: The external torque to use.
        num_simulations: The number of simulations to run.
        parallel_bool: If True, runs the simulations in parallel. If False, runs the simulations sequentially.
        save_trials: If True, saves each trial data to a separate .csv file. Uses more memory, but allows you to see the results of each trial at a later date.
    Outputs:
        None. It writes to the output file.
    """

    world_size = 10

    # convert controller to string
    controller_name = str(controller_type)

    # Create the output file and handle existing files
    output_csv_file = os.path.dirname(__file__) + f'/data/summary_{controller_name}.csv'

    if os.path.exists(output_csv_file):
        os.remove(output_csv_file)
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
        writer.writerow(['traj_number'] + ['pos_tracking_error'] + ['heading_error'] )

    # Create world
    world = World.empty([-world_size/2, world_size/2, -world_size/2, world_size/2, -world_size/2, world_size/2])

    # Generate the data
    start_time = time.time()
    generate_data(output_csv_file, world, vehicle, controllers, wind_profiles,
                   trajectories,num_simulations, ext_force, ext_torque,
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
    
    avg_pos_error = df['pos_tracking_error'][pos_tracking_error_success].mean()
    avg_heading_error = df['heading_error'][pos_tracking_error_success].mean()
    
    print("--------------------------------")
    print(f"Controller: {controller_name}")
    print("--------------------------------")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Average pos_tracking_error: {avg_pos_error:.2f} m")
    print(f"Average heading_error: {avg_heading_error:.2f} deg")
    print("--------------------------------")

    # Update stats CSV
    update_stats_csv(
        controller_name,
        success_rate,
        avg_pos_error,
        avg_heading_error,
        stats_file=os.path.dirname(__file__)+ f'/data/controller_stats.csv'
    )

    return None

def update_stats_csv(controller_name, success_rate, avg_pos_error, avg_heading_error, stats_file='data/controller_stats.csv'):
    """
    Updates or adds controller statistics to the CSV file.
    If controller exists, overwrites its stats. If not, adds a new row.
    """
    import pandas as pd
    from pathlib import Path

    # Create stats directory if it doesn't exist
    Path(stats_file).parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame with new stats
    new_stats = pd.DataFrame({
        'controller': [controller_name],
        'success_rate': [success_rate],
        'avg_position_error': [avg_pos_error],
        'avg_heading_error': [avg_heading_error],
        'last_updated': [pd.Timestamp.now()]
    })

    try:
        # Try to read existing stats
        if Path(stats_file).exists():
            stats_df = pd.read_csv(stats_file)
            # Remove old entry for this controller if it exists
            stats_df = stats_df[stats_df['controller'] != controller_name]
            # Append new stats
            stats_df = pd.concat([stats_df, new_stats], ignore_index=True)
        else:
            stats_df = new_stats

        # Save updated stats
        stats_df.to_csv(stats_file, index=False)
        print(f"\nUpdated statistics in {stats_file}")
        print(stats_df.to_string(index=False))

    except Exception as e:
        print(f"Error updating stats file: {e}")

def main():
    args = parse_args()
    vehicle = Multirotor(quad_params)
    
    # Create randomization config based on experiment type
    config = RandomizationConfig.from_experiment_type(
        args.experiment,
        args.num_trials,
        args.seed
    )
    
    # Generate all randomized components
    trajectories = config.create_trajectories()
    controllers = config.create_controllers(args.controller, quad_params, switch_controller)
    wind_profiles = config.create_wind_profiles()
    ext_force = config.create_ext_force()
    ext_torque = config.create_ext_torque()
    
    parallel = args.controller != 'xadap' and 'mpc' not in args.controller
    
    generate_summary(
        args.controller,
        controllers,
        vehicle,
        wind_profiles,
        trajectories,
        ext_force,
        ext_torque,
        args.num_trials,
        parallel,
        args.save_trials
    )

if __name__ == '__main__':
    main()