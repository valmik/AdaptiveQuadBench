from rotorpy.trajectories.minsnap import MinSnap
from rotorpy.environments import Environment
from rotorpy.utils.occupancy_map import OccupancyMap

import numpy as np                  # For array creation/manipulation
import matplotlib.pyplot as plt     # For plotting, although the simulator has a built in plotter
from scipy.spatial.transform import Rotation  # For doing conversions between different rotation descriptions, applying rotations, etc. 
import os                           # For path generation

import multiprocessing
import csv

from tqdm import tqdm

import time
import datetime


def compute_cost(sim_result):
    """
    Computes the cost from the output of a simulator instance.
    Inputs:
        sim_result: The output of a simulator instance.
    Outputs:
        cost: The cost of the trajectory.
    """

    # Some useful values from the trajectory. 

    time = sim_result['time']
    x = sim_result['state']['x']                                    # Position
    v = sim_result['state']['v']                                    # Velocity
    q = sim_result['state']['q']                                    # Attitude
    w = sim_result['state']['w']                                    # Body rates
    rotor_speeds = sim_result['state']['rotor_speeds']              # Rotor speeds

    x_des = sim_result['flat']['x']                                 # Desired position
    v_des = sim_result['flat']['x_dot']                             # Desired velocity
    yaw_des = sim_result['flat']['yaw']                             # Desired yaw angle
    q_des = sim_result['control']['cmd_q']                          # Desired attitude
    rotor_speeds_des = sim_result['control']['cmd_motor_speeds']    # Desired rotor speeds 

    # Write your cost function here. RMSE of position error over the trajectory
    sim_cost = np.sqrt(np.mean(np.sum((x-x_des)**2, axis=1)))
    # Heading error - convert to euler angles
    euler_angles = Rotation.from_quat(q).as_euler('xyz', degrees=False)
    # absolute heading error over the trajectory
    heading_error = (np.abs(euler_angles[:, 2] - yaw_des)).mean()

    heading_error = np.rad2deg(heading_error)

    return sim_cost, heading_error

def write_to_csv(output_file, row):
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    return None

def single_traj_instance(world, vehicles, controllers, wind_profiles, trajectories,
                         ext_force=None, ext_torque=None,
                         seed=None, save_trial=False, save_trial_path=None):
    """
    Generate a single instance of the simulator with a trajectory. 
    Now accepts a list of vehicles and uses the corresponding one for each trial.
    """
    if seed is not None:
        np.random.seed(seed)
        traj_id = seed
    else:
        np.random.seed()
        traj_id = np.random.randint(0, 2**16)

    # Get the corresponding vehicle, controller and wind profile for this trial
    vehicle = vehicles[traj_id % len(vehicles)]
    controller = controllers[traj_id % len(controllers)]
    wind_profile = wind_profiles[traj_id % len(wind_profiles)]
    traj = trajectories[traj_id % len(trajectories)]
    if ext_force is not None:
        ext_force = ext_force[traj_id % len(ext_force)]
    if ext_torque is not None:
        ext_torque = ext_torque[traj_id % len(ext_torque)]

    controller.update_trajectory(traj)

    # Now create an instance of the simulator and run it. 
    sim_instance = Environment(vehicle=vehicle, controller=controller, 
                             wind_profile=wind_profile,
                             trajectory=traj, sim_rate=100,
                             ext_force=ext_force, ext_torque=ext_torque)

    # Set the initial state to the first waypoint at hover. 
    x0 = {'x': np.array([0, 0, 0]),
          'v': np.zeros(3,),
          'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
          'w': np.zeros(3,),
          'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
          'rotor_speeds': np.array([0,0,0,0])}
    
    sim_instance.vehicle.initial_state = x0

    # Now run the simulator for the length of the trajectory. 
    sim_result = sim_instance.run(t_final = 5, 
                              use_mocap=False, 
                              terminate=False, 
                              plot=False, 
                              plot_mocap=False, 
                              plot_estimator=False, 
                              plot_imu=False, 
                              animate_bool=False, 
                              animate_wind=False, 
                              verbose=False)
    
    if save_trial:
        savepath = save_trial_path
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        sim_instance.save_to_csv(os.path.join(savepath, 'trial_{}.csv'.format(traj_id)))
    
    # Compute the cost of the trajectory from result
    trajectory_cost, heading_error = compute_cost(sim_result)

    summary_output = np.concatenate((np.array([int(traj_id)]), np.array([trajectory_cost]), np.array([heading_error])))

    return summary_output

def generate_data(output_csv_file, world, vehicles, controllers, wind_profiles, trajectories,
                  num_simulations, ext_force=None, ext_torque=None, parallel=True,
                  save_individual_trials=False, save_trial_path=None):
    """
    Generates data. Now accepts a list of vehicles.
    """
    if not parallel:
        for i in tqdm(range(num_simulations), desc="Running simulations (sequentially)..."):
            result = single_traj_instance(world, vehicles, controllers, wind_profiles, trajectories,
                                        ext_force=ext_force, ext_torque=ext_torque,
                                        seed=i, save_trial=save_individual_trials, 
                                        save_trial_path=save_trial_path)
            write_to_csv(output_csv_file, result)
    else:
        # Use multiprocessing to run multiple simulations in parallel.
        num_cores = min(multiprocessing.cpu_count(), 20)

        print("Running {} simulations in parallel with up to {} cores.".format(num_simulations, num_cores))
              
        pool = multiprocessing.Pool(num_cores)

        # Use numpy random to generate seeds for each simulation.
        seeds = np.random.choice(np.arange(num_simulations), size=num_simulations, replace=False)

        manager = multiprocessing.Manager()
        
        lock = manager.Lock()

        def update_results(result):
            with lock:
                write_to_csv(output_file=output_csv_file, row=result)

        code_rate = 1.33  # simulations per second, emperically determined from our machine, but will differ for yours.
        expected_duration_seconds = num_simulations/code_rate  
        
        current_time = datetime.datetime.now()
        end_time = current_time + datetime.timedelta(seconds=expected_duration_seconds)

        print(f"Start time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Expected duration: %3.2f seconds (%3.2f minutes, or %3.2f hours)" % (expected_duration_seconds, expected_duration_seconds/60, expected_duration_seconds/3600))
        print(f"Program *may* end around: {end_time.strftime('%Y-%m-%d %H:%M:%S')}, depending on your machine specs, number of waypoints, distance between waypoints, etc.")

        print("Running simulations (in parallel)...")
        for i in range(num_simulations):
            pool.apply_async(single_traj_instance, 
                           args=(world, vehicles, controllers, wind_profiles, trajectories,
                                 ext_force, ext_torque,
                                 i, save_individual_trials, save_trial_path),
                           callback=update_results)
            
        pool.close()
        pool.join()

    return None
