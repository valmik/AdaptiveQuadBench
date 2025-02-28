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
from quad_param.quadrotor import quad_params
from rotorpy.trajectories.random_motion_prim_traj import RapidTrajectory

from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.world import World
from rotorpy.environments import Environment
from rotorpy.wind.dryden_winds import DrydenGust

from utils.parallel_data_collection import generate_data,compute_cost

import time
import os
import csv
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from randomization_config import RandomizationConfig, ExperimentType
import seaborn as sns
from matplotlib import rcParams
from utils.plotting_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', type=str, nargs='+', default=['geo'], 
                       help='controller types: geo, geo-a, l1geo, l1mpc, indi-a, xadap, mpc, all')
    parser.add_argument('--experiment', type=str, default='no', 
                       choices=[e.value for e in ExperimentType],
                       help='experiment type: no, wind, uncertainty, force, torque, rotoreff')
    parser.add_argument('--num_trials', type=int, default=100, help='number of trials to run')
    parser.add_argument('--seed', type=int, default=42, 
                       help='seed for random number generator')
    parser.add_argument('--save_trials', action='store_true',
                       help='save individual trials to csv')
    parser.add_argument('--serial', action='store_true',
                       help='run in serial')
    parser.add_argument('--vis', action='store_true',
                       help='visualize single trial without saving data')
    parser.add_argument('--when2fail', action='store_true',
                       help='find failure point by increasing disturbance intensity')
    parser.add_argument('--max_intensity', type=float, default=10.0,
                       help='maximum intensity multiplier for when2fail mode')
    parser.add_argument('--intensity_step', type=float, default=1,
                       help='intensity increment step for when2fail mode')
    parser.add_argument('--trajectory', type=str, default='random',
                       choices=['random', 'hover', 'circle'],
                       help='trajectory type to use')
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

def generate_summary(controller_type, controllers, vehicles, wind_profiles, trajectories, ext_force, ext_torque,
                      num_simulations=100, parallel_bool=True, save_trials=False, experiment_type='no', output_file=None):
    """
    Main function for generating data.
    Now accepts a list of vehicles with randomized parameters.
    """
    world_size = 10

    # convert controller to string
    controller_name = str(controller_type)

    # Create the output file and handle existing files
    output_csv_file = output_file if output_file else os.path.dirname(__file__) + f'/data/summary_{controller_name}_{experiment_type}.csv'

    if os.path.exists(output_csv_file):
        os.remove(output_csv_file)
    savepath = None
    if save_trials:
        savepath = os.path.dirname(__file__)+ f'/data/trial_data_{controller_name}_{experiment_type}'
        print(f"savepath: {savepath}")
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        else:
            # Ask the user if they want to remove the existing files in the directory.
            user_input = input(f"The directory {savepath} already exists. Do you want to remove the existing files? (y/n)")
            if user_input == 'y':
                for file in os.listdir(savepath):
                    os.remove(os.path.join(savepath, file))
            elif user_input == 'n':
                raise Exception(f"Please delete or rename the files in the directory {savepath} before running this script.")
            else:
                raise Exception("Invalid input. Please enter 'y' or 'n'.")

    # Append headers to the output file
    with open(output_csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['traj_number'] + ['pos_tracking_error'] + ['heading_error'])

    # Create world
    world = World.empty([-world_size/2, world_size/2, -world_size/2, world_size/2, -world_size/2, world_size/2])

    # Generate the data
    start_time = time.time()
    generate_data(output_csv_file, world, vehicles, controllers, wind_profiles,
                   trajectories, num_simulations, ext_force, ext_torque,
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

    # Update stats CSV with experiment type
    update_stats_csv(
        controller_name,
        experiment_type,
        success_rate,
        avg_pos_error,
        avg_heading_error,
        stats_file=os.path.dirname(__file__)+ f'/data/controller_stats.csv'
    )

    return None

def update_stats_csv(controller_name, experiment_type, success_rate, avg_pos_error, avg_heading_error, stats_file='data/controller_stats.csv'):
    """
    Updates or adds controller statistics to the CSV file.
    Now includes experiment type in the tracking.
    """
    import pandas as pd
    from pathlib import Path

    # Create stats directory if it doesn't exist
    Path(stats_file).parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame with new stats
    new_stats = pd.DataFrame({
        'controller': [controller_name],
        'experiment': [experiment_type],
        'success_rate': [success_rate],
        'avg_position_error': [avg_pos_error],
        'avg_heading_error': [avg_heading_error],
        'last_updated': [pd.Timestamp.now()]
    })
    try:
        # Try to read existing stats
        if Path(stats_file).exists():
            stats_df = pd.read_csv(stats_file)
            # Remove old entry for this controller+experiment combination if it exists
            stats_df = stats_df[~((stats_df['controller'] == controller_name) & 
                                (stats_df['experiment'] == experiment_type))]
            # Append new stats
            stats_df = pd.concat([stats_df, new_stats], ignore_index=True)
        else:
            stats_df = new_stats

        # Save updated stats
        stats_df.to_csv(stats_file, index=False)
        print(f"\nUpdated statistics in {stats_file}")
        print(stats_df.to_string(index=False))

    except Exception as exp:
        print(f"Error updating stats file: {exp}")

def visualize_trials(experiment_type, vehicles, controllers, controller_types, wind_profiles, trajectory, ext_force=None, ext_torque=None):
    """Run trials for multiple controllers and create comparison plots"""
    sim_results = []
    controller_palette = sns.color_palette("husl", len(controller_types))
    disturbance_palette = sns.color_palette("Set1", 3)
    
    # Run simulation for each controller
    for vehicle, controller in zip(vehicles, controllers):
        if 'ModelPredictiveControl' in str(controller.__class__):
            controller.update_trajectory(trajectory)
        sim_instance = Environment(
            vehicle=vehicle, 
            controller=controller,
            wind_profile=wind_profiles[0] if wind_profiles[0] is not None else None,
            trajectory=trajectory,
            sim_rate=100,
            ext_force=ext_force[0] if ext_force is not None else None,
            ext_torque=ext_torque[0] if ext_torque is not None else None
        )

        x0 = {'x': np.array([0, 0, 0]),
              'v': np.zeros(3,),
              'q': np.array([0, 0, 0, 1]),
              'w': np.zeros(3,),
              'wind': np.array([0,0,0]),
              'rotor_speeds': np.array([0,0,0,0])}

        # Run simulation without built-in visualization
        sim_result = sim_instance.run(
            t_final=5,
            use_mocap=False,
            terminate=False,
            plot=False,  # Disable default plotting
            animate_bool=False,  # Disable default animation
            verbose=False
        )
        sim_results.append(sim_result)

    if experiment_type != 'rotoreff':
        fig = plt.figure(figsize=(6, 8))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1])
        
        ax1 = fig.add_subplot(gs[0], projection='3d')
        plot_3d_trajectory(ax1, sim_results, controller_types, controller_palette)
        
        ax2 = fig.add_subplot(gs[1])
        plot_position_error(ax2, sim_results, controller_types, controller_palette)
        
        ax3 = fig.add_subplot(gs[2])
        plot_disturbance(ax3, sim_results[0], experiment_type, disturbance_palette)
        
        ax4 = fig.add_subplot(gs[3])
        plot_motor_speeds(ax4, sim_results, controller_types, controller_palette)
    else:
        fig = plot_rotor_efficiency_comparison(sim_results, controller_types)
    
    plt.tight_layout(h_pad=1.0, w_pad=1.0)
    plot_dir = os.path.dirname(__file__)+ f'/data/plots'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f'{plot_dir}/vis_{experiment_type}_{"_".join(controller_types)}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

    # Print performance metrics for each controller
    print("\nPerformance Metrics:")
    print("-" * 50)
    for result, ctrl_type in zip(sim_results, controller_types):
        pos_error, heading_error = compute_cost(result)
        print(f"\nController: {ctrl_type}")
        print(f"Average position error: {pos_error:.3f} m")
        print(f"Average heading error: {heading_error:.3f} deg")
    print("-" * 50)

    return sim_results

def run_when2fail(args, controllers_to_run):
    """Run experiments with increasing disturbance intensity until failure or max intensity"""
    
    # Initialize tracking variables
    intensities = {ctrl:[] for ctrl in controllers_to_run}
    success_rates = {ctrl: [] for ctrl in controllers_to_run}
    pos_errors = {ctrl: [] for ctrl in controllers_to_run}
    heading_errors = {ctrl: [] for ctrl in controllers_to_run}
    
    
    print(f"\nRunning when2fail analysis for experiment type: {args.experiment}")
    
    # Create base config and get constant components
    config = RandomizationConfig.from_experiment_type(
        args.experiment,
        args.num_trials,
        quad_params,
        args.seed,
        trajectory_type=args.trajectory
    )
    base_components = config.create_base_components()
    
    # Start with original intensity
    intensity = args.intensity_step
    active_controllers = controllers_to_run.copy()
    
    while intensity <= args.max_intensity and active_controllers:
        print(f"\nTesting intensity multiplier: {intensity:.1f}")
        print(f"Active controllers: {active_controllers}")
        
        # Scale ranges for current intensity
        config.scale_ranges_with_intensity(intensity)
        
        # Generate intensity-dependent components
        varied_components = config.create_varied_components()
        
        # Create vehicles from parameters
        vehicles = [Multirotor(params) for params in 
                   (varied_components.get('vehicle_params') or base_components['vehicle_params'])]
        # Create controllers using appropriate parameters
        controller_params = varied_components.get('controller_params') or base_components['controller_params']
        
        # Test each active controller
        controllers_to_remove = []
        for controller_type in active_controllers:
            print(f"Testing {controller_type}...")
            
            controllers = [
                switch_controller(controller_type, params) 
                for params in controller_params
            ]
            
            # Create temporary CSV for this run
            temp_csv = f'data/temp_{controller_type}_{intensity:.1f}.csv'
            
            use_parallel = (not args.serial) and controller_type != 'xadap'
            generate_summary(
                controller_type,
                controllers,
                vehicles,
                varied_components['wind_profiles'],
                base_components['trajectories'],  # Use constant trajectories
                varied_components['ext_force'],
                varied_components['ext_torque'],
                args.num_trials,
                use_parallel,
                False,
                args.experiment,
                output_file=temp_csv
            )
            
            # Read results
            df = pd.read_csv(temp_csv)
            success_rate = (df['pos_tracking_error'] < 5).mean() * 100
            avg_pos_error = df['pos_tracking_error'].mean()
            avg_heading_error = df['heading_error'].mean()
            
            # Store results
            intensities[controller_type].append(intensity)
            success_rates[controller_type].append(success_rate)
            pos_errors[controller_type].append(avg_pos_error)
            heading_errors[controller_type].append(avg_heading_error)
            
            # Check if controller has failed (0% success rate)
            if success_rate == 0:
                controllers_to_remove.append(controller_type)
                print(f"{controller_type} failed at intensity {intensity:.1f}")
            
            # Clean up temporary file
            os.remove(temp_csv)
        
        # Remove failed controllers
        for ctrl in controllers_to_remove:
            active_controllers.remove(ctrl)
        
        # Increment intensity
        intensity += args.intensity_step
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Success Rate Plot
    plt.subplot(131)
    for ctrl in controllers_to_run:
        plt.plot(intensities[ctrl], 
                success_rates[ctrl], marker='o', label=ctrl)
    plt.xlabel('Disturbance Intensity Multiplier')
    plt.ylabel('Success Rate (\%)')
    plt.title('Success Rate vs Disturbance Intensity')
    plt.grid(True)
    plt.legend()
    
    # Position Error Plot
    plt.subplot(132)
    for ctrl in controllers_to_run:
        plt.plot(intensities[ctrl], 
                pos_errors[ctrl], marker='o', label=ctrl)
    plt.xlabel('Disturbance Intensity Multiplier')
    plt.ylabel('Average Position Error (m)')
    plt.title('Position Error vs Disturbance Intensity')
    plt.grid(True)
    plt.legend()
    
    # Heading Error Plot
    plt.subplot(133)
    for ctrl in controllers_to_run:
        plt.plot(intensities[ctrl], 
                heading_errors[ctrl], marker='o', label=ctrl)
    plt.xlabel('Disturbance Intensity Multiplier')
    plt.ylabel('Average Heading Error (deg)')
    plt.title('Heading Error vs Disturbance Intensity')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plot_dir = os.path.dirname(__file__)+ f'/data/plots'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f'{plot_dir}/vis_when2fail_{args.experiment}_{"_".join(controllers_to_run)}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\nWhen2Fail Analysis Summary:")
    print("-" * 50)
    for ctrl in controllers_to_run:
        failure_intensity = args.intensity_step * len(success_rates[ctrl])
        print(f"\n{ctrl}:")
        print(f"Failure intensity: {failure_intensity:.1f}")
        print(f"Final success rate: {success_rates[ctrl][-1]:.1f}%")
        print(f"Final position error: {pos_errors[ctrl][-1]:.2f} m")
        print(f"Final heading error: {heading_errors[ctrl][-1]:.2f} deg")

def main():
    args = parse_args()
    
    # Define all available controllers
    all_controllers = ['geo', 'geo-a', 'l1geo', 'indi-a', 'l1mpc', 'mpc', 'xadap']
    
    # If 'all' is in the controller list, use all controllers
    if 'all' in args.controller:
        controllers_to_run = all_controllers
    else:
        controllers_to_run = args.controller

    ModifyPlotForPublication()
    if args.when2fail:
        run_when2fail(args, controllers_to_run)
    elif args.vis:
        # Create world
        world_size = 10
        world = World.empty([-world_size/2, world_size/2, -world_size/2, world_size/2, -world_size/2, world_size/2])

        # Override num_trials if visualization is enabled
        args.num_trials = 1
        print("Visualization mode: Running single trial with multiple controllers")

        # Create randomization config
        config = RandomizationConfig.from_experiment_type(
            args.experiment,
            args.num_trials,
            quad_params,
            args.seed,
            trajectory_type=args.trajectory
        )

        # Generate all randomized components
        trajectories = config.create_trajectories()
        wind_profiles = config.create_wind_profiles()
        ext_force = config.create_ext_force()
        ext_torque = config.create_ext_torque()
        vehicle_params_list = config.create_vehicle_params(quad_params)
        vehicles = [Multirotor(params) for params in vehicle_params_list]
        controller_params_list = config.create_controller_params(quad_params)

        # Create one controller of each type using the same parameters
        controllers = [
            switch_controller(ctrl_type, controller_params_list[0]) 
            for ctrl_type in controllers_to_run
        ]
        # Run visualization with all controllers
        sim_results = visualize_trials(
            args.experiment,
            [vehicles[0]] * len(controllers),  # Use same vehicle for all controllers
            controllers,
            controllers_to_run,
            wind_profiles,
            trajectories[0],
            ext_force,
            ext_torque
        )
    else:
        # Create randomization config
        config = RandomizationConfig.from_experiment_type(
            args.experiment,
            args.num_trials,
            quad_params,
            args.seed,
            trajectory_type=args.trajectory
        )

        # Generate all randomized components once
        trajectories = config.create_trajectories()
        wind_profiles = config.create_wind_profiles()
        ext_force = config.create_ext_force()
        ext_torque = config.create_ext_torque()
        vehicle_params_list = config.create_vehicle_params(quad_params)
        vehicles = [Multirotor(params) for params in vehicle_params_list]
        controller_params_list = config.create_controller_params(quad_params)

        # Run normal batch of trials for each controller
        for controller_type in controllers_to_run:
            print(f"\nRunning experiments for controller: {controller_type}")
            print("=" * 50)
            
            controllers = [
                switch_controller(controller_type, params) 
                for params in controller_params_list
            ]
            
            use_parallel = (not args.serial) and controller_type != 'xadap'
            generate_summary(
                controller_type,
                controllers,
                vehicles,
                wind_profiles,
                trajectories,
                ext_force,
                ext_torque,
                args.num_trials,
                use_parallel,
                args.save_trials,
                args.experiment
            )

if __name__ == '__main__':
    main()