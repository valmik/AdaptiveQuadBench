import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from experiments.plotting_utils import *
from utils.parallel_data_collection import compute_cost
from randomization_config import ExperimentType

class ExperimentVisualizer:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.plot_dir = self.data_dir / 'plots'
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        ModifyPlotForPublication()

    def visualize_trials(self, experiment_type, sim_results, controller_types, controller_param, vehicle_params):
        """Create plots based on experiment type
        
        Args:
            experiment_type (str): Type of experiment ('wind', 'force', 'torque', 'uncertainty', 'rotoreff', 'payload')
            sim_results (list): List of simulation results
            controller_types (list): List of controller names
            controller_param (list): List of controller parameters
            
        Raises:
            ValueError: If experiment_type is not supported
        """

        controller_palette = sns.color_palette("husl", len(controller_types))
        disturbance_palette = sns.color_palette("Set1", 3)
        
        plot_functions = {
            'wind': self._plot_wind_experiment,
            'force': self._plot_force_experiment,
            'torque': self._plot_torque_experiment,
            'uncertainty': self._plot_uncertainty_experiment,
            'rotoreff': self._plot_rotor_efficiency_experiment,
            'payload': self._plot_payload_experiment
        }
        
        if experiment_type not in plot_functions:
            raise ValueError(f"Unsupported experiment type: {experiment_type}. "
                            f"Must be one of {list(plot_functions.keys())}")
        
        fig = plot_functions[experiment_type](sim_results, controller_types, controller_param, vehicle_params,
                                            controller_palette, disturbance_palette)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / f'vis_{experiment_type}_{"_".join(controller_types)}.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()


        self._print_metrics(sim_results, controller_types)
        
        return sim_results

    def _plot_wind_experiment(self, sim_results, controller_types, controller_param, vehicle_params, controller_palette, disturbance_palette):
        """Specific plotting for wind experiments"""
        fig = plt.figure(figsize=(8,4))
        gs = fig.add_gridspec(2,4)
        
        # 3D trajectory and wind vector plot
        ax1 = fig.add_subplot(gs[0:, 0:2], projection='3d')
        plot_3d_trajectory(ax1, sim_results, controller_types, controller_palette)
        
        # Wind components
        ax2 = fig.add_subplot(gs[1, 2:])
        plot_disturbance(ax2, sim_results[0], 'wind', disturbance_palette)
        ax2.set_ylabel('Wind [m/s]')
        # Position tracking error
        ax3 = fig.add_subplot(gs[0, 2:])
        plot_position_error(ax3, sim_results, controller_types, controller_palette)
        
        
        return fig
    
    def _plot_force_experiment(self, sim_results, controller_types, controller_param, vehicle_params, controller_palette, disturbance_palette):
        """Specific plotting for force experiments"""
        fig = plt.figure(figsize=(8,4))
        gs = fig.add_gridspec(2,4)
        
        # 3D trajectory and force vector plot
        ax1 = fig.add_subplot(gs[0:, 0:2], projection='3d')
        plot_3d_trajectory(ax1, sim_results, controller_types, controller_palette)
        
        # Force components
        ax2 = fig.add_subplot(gs[1, 2:])
        plot_disturbance(ax2, sim_results[0], 'force', disturbance_palette)
        ax2.set_ylabel('Force [N]')
        # Position tracking error
        ax3 = fig.add_subplot(gs[0, 2:])
        plot_position_error(ax3, sim_results, controller_types, controller_palette)
        
        
        return fig
    
    def _plot_torque_experiment(self, sim_results, controller_types, controller_param, vehicle_params, controller_palette, disturbance_palette):
        """Specific plotting for torque experiments"""
        fig = plt.figure(figsize=(8,4))
        gs = fig.add_gridspec(2,4)
        
        # 3D trajectory and torque vector plot
        ax1 = fig.add_subplot(gs[0:, 0:2], projection='3d')
        plot_3d_trajectory(ax1, sim_results, controller_types, controller_palette)
        
        # Torque components
        ax2 = fig.add_subplot(gs[1, 2:])
        plot_disturbance(ax2, sim_results[0], 'torque', disturbance_palette)
        ax2.set_ylabel('Torque [Nm]')
        # Position tracking error
        ax3 = fig.add_subplot(gs[0, 2:])
        plot_position_error(ax3, sim_results, controller_types, controller_palette)

        return fig

    def _plot_rotor_efficiency_experiment(self, sim_results, controller_types, controller_param, vehicle_params, controller_palette, disturbance_palette):
        num_controllers = len(controller_types)
        fig = plt.figure(figsize=(8,6*num_controllers))
        each_row_num = 3
        gs = fig.add_gridspec(each_row_num*num_controllers, 4)

        # Get global min/max values
        pos_error_min, pos_error_max = get_position_error_range(sim_results)
        rotor_speed_min, rotor_speed_max = get_rotor_speed_range(sim_results)
        window = np.ones(20)/20  # 20-point moving average

        for idx, (result, ctrl_type) in enumerate(zip(sim_results, controller_types)):
            ax1 = fig.add_subplot(gs[idx*each_row_num:(idx+1)*each_row_num, 0:2], projection='3d')
            controller_palette = sns.color_palette("husl", len(controller_types))
            plot_3d_trajectory(ax1, sim_results, controller_types, controller_palette)
            plot_position_errors_subplot(fig, gs[idx*each_row_num, 2:], result, ctrl_type, idx==0,
                                    pos_error_min, pos_error_max)
            plot_rotor_speeds_subplot(fig, gs[idx*each_row_num+1,2:], result, ctrl_type, idx==0,
                                    rotor_speed_min, rotor_speed_max, window)
            # plot the rotor efficiency
            ax3 = fig.add_subplot(gs[idx*each_row_num+2, 2:])
            rotor_eff = np.tile(vehicle_params['rotor_efficiency'], (result['time'].shape[0], 1))
            ax3.plot(result['time'], rotor_eff[:,0], label='Rotor 1')
            ax3.plot(result['time'], rotor_eff[:,1], label='Rotor 2')
            ax3.plot(result['time'], rotor_eff[:,2], label='Rotor 3')
            ax3.plot(result['time'], rotor_eff[:,3], label='Rotor 4')
            ax3.legend()
            ax3.grid(True)
            ax3.set_xlabel('Time [s]')
            ax3.set_ylabel('Rotor Efficiency')
            
        return fig
    
    def _plot_uncertainty_experiment(self, sim_results, controller_types, controller_param, vehicle_params, controller_palette, disturbance_palette):
        """Specific plotting for uncertainty experiments"""
        """
        Create a radar chart comparing ground truth and reference model parameters
        
        Args:
            ground_truth_params (dict): Ground truth parameters
            reference_params (dict): Reference model parameters
        """
        # Select key parameters to compare
        fig = plt.figure(figsize=(8,4))
        gs = fig.add_gridspec(2,4)
        ax1 = fig.add_subplot(gs[0:, 0:2], projection='3d')
        plot_3d_trajectory(ax1, sim_results, controller_types, controller_palette)
        ax1.set_title('3D Trajectory')
        ax2 = fig.add_subplot(gs[0:, 2:], projection='polar')
        plot_model_uncertainty(ax2, controller_param, vehicle_params)

        return fig
    
    
    

    def _plot_payload_experiment(self, sim_results, controller_types, controller_param, vehicle_params, controller_palette, disturbance_palette):
        """Specific plotting for payload experiments"""
        fig = plt.figure(figsize=(6,6))
        gs = fig.add_gridspec(4,4)
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(gs[0:2, 0:2], projection='3d')
        plot_3d_trajectory(ax1, sim_results, controller_types, 
                           controller_palette,elev=45,azim=-90,roll=0)
        
        # Position tracking error
        ax2 = fig.add_subplot(gs[2, :])
        plot_position_error(ax2, sim_results, controller_types, controller_palette)
        
        # Payload mass ratio
        force = sim_results[0]['state']['ext_force']
        torque = sim_results[0]['state']['ext_torque']
        force_mag = np.linalg.norm(force, axis=1)
        payload_mass_ratio = force_mag / (vehicle_params['mass'] * 9.81)
        ax3 = fig.add_subplot(gs[3, :])
        ax3.plot(sim_results[0]['time'], payload_mass_ratio)
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Payload Mass Ratio [\%]')
        ax3.grid(True)

        # Payload location visualization (top-down view)
        ax4 = fig.add_subplot(gs[0:2, 2:], aspect='equal')
        plot_drone(ax4)
        # Compute and plot payload location
        def compute_force_location(f, t):
            """Compute payload location from force and torque"""
            force_mag = np.linalg.norm(f)
            if force_mag < 1e-6:
                return np.zeros(3)
            # Project torque onto plane perpendicular to force
            f_unit = f / force_mag
            t_perp = t - np.dot(t, f_unit) * f_unit
            # Compute location
            r = -np.cross(t_perp, f) / (force_mag ** 2)
            return r

        locations = np.array([compute_force_location(f, t) for f, t in zip(force, torque)])
        # only plot the non-zero locations
        locations = locations[np.linalg.norm(locations, axis=1) > 1e-6]
        ax4.plot(locations[0, 0], locations[0, 1], 'r*', alpha=0.5, label='Payload')
        
        # Formatting
        ax4.set_xlabel('X [m]')
        ax4.set_ylabel('Y [m]')
        ax4.set_title('Payload Location (Top View)')
        ax4.grid(True)
        ax4.legend()
        
        return fig

    def _print_metrics(self, sim_results, controller_types):
        print("\nPerformance Metrics:")
        print("-" * 50)
        for result, ctrl_type in zip(sim_results, controller_types):
            pos_error, heading_error = compute_cost(result)
            print(f"\nController: {ctrl_type}")
            print(f"Average position error: {pos_error:.3f} m")
            print(f"Average heading error: {heading_error:.3f} deg")
        print("-" * 50)

    def plot_when2fail_results(self, experiment_type, controller_types, intensities, 
                             success_rates, pos_errors, heading_errors):
        """Plot the results of the when2fail analysis with physical disturbance values"""
        plt.figure(figsize=(3.5*3.5, 3.5))
        
        # Map intensities to actual physical quantities based on experiment type
        x_labels = {
            ExperimentType.WIND: (
                'Wind Speed (m/s)', 
                lambda i: i
            ),
            ExperimentType.FORCE: (
                'External Force Magnitude (N)', 
                lambda i: i
            ),
            ExperimentType.TORQUE: (
                'External Torque Magnitude (Nm)', 
                lambda i: i
            ),
            ExperimentType.UNCERTAINTY: (
                'Max $|c|$', 
                lambda i: 100 * i
            ),
            ExperimentType.ROTOR_EFFICIENCY: (
                'Max Rotor Efficiency Change (\%)', 
                lambda i: 100 * min(0.6, i)  # Special case as it's symmetric
            ),
            ExperimentType.PAYLOAD: (
                'Payload Mass Ratio (\%)', 
                lambda i: 100 * min(2.0, i)
            )
        }
        xlabel, transform_fn = x_labels.get(ExperimentType(experiment_type), ('Intensity Multiplier', lambda i: i))

        # Success Rate Plot
        plt.subplot(131)
        for ctrl in controller_types:
            x_values = [transform_fn(i) for i in intensities[ctrl]]
            plt.plot(x_values, success_rates[ctrl], marker='o', label=ctrl)
        plt.xlabel(xlabel)
        plt.ylabel('Success Rate (\%)')
        plt.grid(True)
        plt.legend()
        # Position Error Plot
        plt.subplot(132)
        for ctrl in controller_types:
            x_values = [transform_fn(i) for i in intensities[ctrl]]
            plt.plot(x_values, pos_errors[ctrl], marker='o', label=ctrl)
        plt.xlabel(xlabel)
        plt.ylabel('Average Position Error (m)')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        
        # Heading Error Plot
        plt.subplot(133)
        for ctrl in controller_types:
            x_values = [transform_fn(i) for i in intensities[ctrl]]
            plt.plot(x_values, heading_errors[ctrl], marker='o', label=ctrl)
        plt.xlabel(xlabel)
        plt.ylabel('Average Heading Error (deg)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / f'vis_when2fail_{experiment_type}_{"_".join(controller_types)}.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()        
        # Print summary with physical quantities
        print("\nWhen2Fail Analysis Summary:")
        print("-" * 50)
        print(f"Experiment type: {experiment_type}")
        for ctrl in controller_types:
            print(f"\n{ctrl}:")
            print(f"Final success rate: {success_rates[ctrl][-1]:.1f}%")
            print(f"Final position error: {pos_errors[ctrl][-1]:.2f} m")
            print(f"Final heading error: {heading_errors[ctrl][-1]:.2f} deg")
