from typing import List, Optional, Callable
import numpy as np
import time
from pathlib import Path
from quad_param.quadrotor import quad_params
from rotorpy.world import World
from rotorpy.vehicles.multirotor import Multirotor
from randomization_config import RandomizationConfig
from .config_manager import ExperimentConfig
from .results_manager import ResultsManager
from .visualizer import ExperimentVisualizer
import pandas as pd
import os
import matplotlib.pyplot as plt
from rotorpy.environments import Environment

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, controller_factory: Callable):
        self.config = config
        self.controller_factory = controller_factory
        self.results_manager = ResultsManager()
        self.visualizer = ExperimentVisualizer()
        
    def run(self):
        if self.config.when2fail:
            self._run_when2fail()
        elif self.config.visualize:
            self._run_visualization()
        else:
            self._run_normal_experiments()
            
    def _run_normal_experiments(self):
        rand_config = RandomizationConfig.from_experiment_type(
            self.config.experiment_type,
            self.config.num_trials,
            quad_params,
            self.config.seed,
            trajectory_type=self.config.trajectory_type
        )
        
        components = self._generate_components(rand_config)
        
        for controller_type in self.config.controller_types:
            print(f"\nRunning experiments for controller: {controller_type}")
            print("=" * 50)
            
            controllers = [
                self.controller_factory(controller_type, params) 
                for params in components['controller_params']
            ]
            
            use_parallel = (not self.config.use_serial) and controller_type != 'xadap'
            self.results_manager.generate_summary(
                controller_type,
                controllers,
                components['vehicles'],
                components['wind_profiles'],
                components['trajectories'],
                components['ext_force'],
                components['ext_torque'],
                self.config.num_trials,
                use_parallel,
                self.config.save_trials,
                self.config.experiment_type
            )

    def _generate_components(self, config):
        trajectories = config.create_trajectories()
        wind_profiles = config.create_wind_profiles()
        ext_force = config.create_ext_force()
        ext_torque = config.create_ext_torque()
        vehicle_params_list = config.create_vehicle_params(quad_params)
        vehicles = [Multirotor(params) for params in vehicle_params_list]
        controller_params_list = config.create_controller_params(quad_params)
        
        return {
            'trajectories': trajectories,
            'wind_profiles': wind_profiles,
            'ext_force': ext_force,
            'ext_torque': ext_torque,
            'vehicles': vehicles,
            'controller_params': controller_params_list
        }

    def _run_visualization(self):
        print("Visualization mode: Running single trial with multiple controllers")
        
        rand_config = RandomizationConfig.from_experiment_type(
            self.config.experiment_type,
            1,  # Single trial for visualization
            quad_params,
            self.config.seed,
            trajectory_type=self.config.trajectory_type
        )
        
        components = self._generate_components(rand_config)
        controllers = [
            self.controller_factory(ctrl_type, components['controller_params'][0]) 
            for ctrl_type in self.config.controller_types
        ]
        
        # Run simulations and collect results
        sim_results = []
        for vehicle, controller in zip([components['vehicles'][0]] * len(controllers), controllers):
            if 'ModelPredictiveControl' in str(controller.__class__):
                controller.update_trajectory(components['trajectories'][0])
            
            sim_instance = Environment(
                vehicle=vehicle, 
                controller=controller,
                wind_profile=components['wind_profiles'][0] if components['wind_profiles'][0] is not None else None,
                trajectory=components['trajectories'][0],
                sim_rate=100,
                ext_force=components['ext_force'][0] if components['ext_force'] is not None else None,
                ext_torque=components['ext_torque'][0] if components['ext_torque'] is not None else None
            )

            x0 = {
                'x': np.array([0, 0, 0]),
                'v': np.zeros(3,),
                'q': np.array([0, 0, 0, 1]),
                'w': np.zeros(3,),
                'wind': np.array([0,0,0]),
                'rotor_speeds': np.array([0,0,0,0])
            }

            sim_result = sim_instance.run(
                t_final=5,
                use_mocap=False,
                terminate=False,
                plot=False,
                animate_bool=False,
                verbose=False
            )
            sim_results.append(sim_result)
        
        # Visualize results
        self.visualizer.visualize_trials(
            experiment_type=self.config.experiment_type,
            sim_results=sim_results,
            controller_types=self.config.controller_types
        )

    def _run_when2fail(self):
        """Run experiments with increasing disturbance intensity until failure or max intensity"""

        # Initialize tracking variables
        intensities = {ctrl: [] for ctrl in self.config.controller_types}
        success_rates = {ctrl: [] for ctrl in self.config.controller_types}
        pos_errors = {ctrl: [] for ctrl in self.config.controller_types}
        heading_errors = {ctrl: [] for ctrl in self.config.controller_types}
        
        print(f"\nRunning when2fail analysis for experiment type: {self.config.experiment_type}")
        
        # Create base config and get constant components
        rand_config = RandomizationConfig.from_experiment_type(
            self.config.experiment_type,
            self.config.num_trials,
            quad_params,
            self.config.seed,
            trajectory_type=self.config.trajectory_type
        )
        base_components = rand_config.create_base_components()
        
        # Start with original intensity
        intensity = 0
        active_controllers = self.config.controller_types.copy()
        
        while intensity <= self.config.max_intensity and active_controllers:
            print(f"\nTesting intensity multiplier: {intensity:.1f}")
            print(f"Active controllers: {active_controllers}")
            
            # Scale ranges for current intensity
            rand_config.scale_ranges_with_intensity(intensity)
            
            # Generate intensity-dependent components
            varied_components = rand_config.create_varied_components()
            
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
                    self.controller_factory(controller_type, params) 
                    for params in controller_params
                ]
                
                # Create temporary CSV for this run
                temp_csv = self.results_manager.data_dir / f'temp_{controller_type}_{intensity:.1f}.csv'
                
                use_parallel = (not self.config.use_serial) and controller_type != 'xadap'
                self.results_manager.generate_summary(
                    controller_type,
                    controllers,
                    vehicles,
                    varied_components['wind_profiles'],
                    base_components['trajectories'],  # Use constant trajectories
                    varied_components['ext_force'],
                    varied_components['ext_torque'],
                    self.config.num_trials,
                    use_parallel,
                    False,
                    self.config.experiment_type,
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
                temp_csv.unlink()
            
            # Remove failed controllers
            for ctrl in controllers_to_remove:
                active_controllers.remove(ctrl)
            
            # Increment intensity
            intensity += self.config.intensity_step
        
        # Visualize results with rand_config
        self.visualizer.plot_when2fail_results(
            experiment_type=self.config.experiment_type,
            controller_types=self.config.controller_types,
            intensities=intensities,
            success_rates=success_rates,
            pos_errors=pos_errors,
            heading_errors=heading_errors,
        )
