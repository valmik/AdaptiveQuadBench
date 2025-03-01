import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from utils.plotting_utils import *
from utils.parallel_data_collection import compute_cost
from randomization_config import ExperimentType

class ExperimentVisualizer:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.plot_dir = self.data_dir / 'plots'
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        ModifyPlotForPublication()

    def visualize_trials(self, experiment_type, sim_results, controller_types):
        """Visualize simulation results for multiple controllers"""
        controller_palette = sns.color_palette("husl", len(controller_types))
        disturbance_palette = sns.color_palette("Set1", 3)
        
        self._create_plots(experiment_type, sim_results, controller_types, 
                          controller_palette, disturbance_palette)
        self._print_metrics(sim_results, controller_types)
        
        return sim_results

    def _create_plots(self, experiment_type, sim_results, controller_types, 
                     controller_palette, disturbance_palette):
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
        plt.savefig(self.plot_dir / f'vis_{experiment_type}_{"_".join(controller_types)}.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()

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
        plt.figure(figsize=(15, 5))
        
        # Map intensities to actual physical quantities based on experiment type
        x_labels = {
            ExperimentType.WIND: (
                'Wind Speed (m/s)', 
                lambda i: i
            ),
            ExperimentType.FORCE: (
                'External Force (N)', 
                lambda i: i
            ),
            ExperimentType.TORQUE: (
                'External Torque (Nm)', 
                lambda i: i
            ),
            ExperimentType.UNCERTAINTY: (
                'Parameter Variation from Nominal (\%)', 
                lambda i: 100 * i
            ),
            ExperimentType.ROTOR_EFFICIENCY: (
                'Rotor Efficiency Change (\%)', 
                lambda i: 100 * min(0.6, i)  # Special case as it's symmetric
            ),
            ExperimentType.PAYLOAD: (
                'Payload Mass Ratio', 
                lambda i: min(2.0, i)
            )
        }
        xlabel, transform_fn = x_labels.get(ExperimentType(experiment_type), ('Intensity Multiplier', lambda i: i))

        # Success Rate Plot
        plt.subplot(131)
        for ctrl in controller_types:
            x_values = [transform_fn(i) for i in intensities[ctrl]]
            plt.plot(x_values, success_rates[ctrl], marker='o', label=ctrl)
        plt.xlabel(xlabel)
        plt.ylabel('Success Rate (%)')
        plt.grid(True)
        plt.legend()
        
        # Position Error Plot
        plt.subplot(132)
        for ctrl in controller_types:
            x_values = [transform_fn(i) for i in intensities[ctrl]]
            plt.plot(x_values, pos_errors[ctrl], marker='o', label=ctrl)
        plt.xlabel(xlabel)
        plt.ylabel('Average Position Error (m)')
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