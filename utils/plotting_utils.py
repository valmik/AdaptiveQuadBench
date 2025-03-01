import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib import rcParams



def ModifyPlotForPublication(text_size=10, tick_size=10, legend_size=10):
    """Set matplotlib parameters for publication quality plots"""
    rcParams['axes.labelsize'] = text_size
    rcParams['axes.formatter.use_mathtext'] = True
    rcParams['xtick.labelsize'] = tick_size
    rcParams['ytick.labelsize'] = tick_size
    rcParams['legend.fontsize'] = legend_size
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['cmr10']
    rcParams['text.usetex'] = True

def plot_3d_trajectory(ax, sim_results, controller_types, controller_palette):
    """Plot 3D trajectory comparison"""
    x_des = sim_results[0]['flat']['x']
    ax.plot(x_des[:,0], x_des[:,1], x_des[:,2], 'k--', label='Desired')
    ax.scatter(x_des[0,0], x_des[0,1], x_des[0,2], color='red', label='Start')
    ax.scatter(x_des[-1,0], x_des[-1,1], x_des[-1,2], color='blue', label='End')
    
    for result, ctrl_type in zip(sim_results, controller_types):
        x = result['state']['x']
        ax.plot(x[:,0], x[:,1], x[:,2], label=ctrl_type, 
                color=controller_palette[controller_types.index(ctrl_type)])
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_zlim(-2,2)
    ax.set_title('3D Trajectory')
    ax.legend(ncol=2, loc='best')

def plot_position_error(ax, sim_results, controller_types, controller_palette):
    """Plot position tracking error norm for all controllers"""
    for result, ctrl_type in zip(sim_results, controller_types):
        time = result['time']
        x = result['state']['x']
        x_des = result['flat']['x']
        pos_error = np.linalg.norm(x - x_des, axis=1)
        ax.plot(time, pos_error, label=ctrl_type, 
                color=controller_palette[controller_types.index(ctrl_type)])
    
    ax.set_ylabel('Position Error [m]')
    ax.legend()
    ax.grid(True)

def plot_disturbance(ax, result, experiment_type, disturbance_palette):
    """Plot disturbance data"""
    time = result['time']
    if experiment_type == 'wind':
        wind = result['state']['wind']
        ax.plot(time, wind[:,0], label='X', color=disturbance_palette[0])
        ax.plot(time, wind[:,1], label='Y', color=disturbance_palette[1])
        ax.plot(time, wind[:,2], label='Z', color=disturbance_palette[2])
        ax.set_ylabel('Wind [m/s]')
    elif experiment_type == 'torque':
        torque = result['state']['ext_torque']
        ax.plot(time, torque[:,0], label='X', color=disturbance_palette[0])
        ax.plot(time, torque[:,1], label='Y', color=disturbance_palette[1])
        ax.plot(time, torque[:,2], label='Z', color=disturbance_palette[2])
        ax.set_ylabel('Torque [Nm]')
    else: # force, payload, uncertainty # ? no good way to visualize single uncertainty
        force = result['state']['ext_force']
        ax.plot(time, force[:,0], label='X', color=disturbance_palette[0])
        ax.plot(time, force[:,1], label='Y', color=disturbance_palette[1])
        ax.plot(time, force[:,2], label='Z', color=disturbance_palette[2])
        ax.set_ylabel('Force [N]')
    ax.legend()
    ax.grid(True)

def plot_motor_speeds(ax, sim_results, controller_types, controller_palette):
    """Plot average motor speeds"""
    for result, ctrl_type in zip(sim_results, controller_types):
        motor_speeds = result['control']['cmd_motor_speeds']
        time = result['time']
        ax.plot(time, motor_speeds.mean(axis=1), label=ctrl_type, 
                color=controller_palette[controller_types.index(ctrl_type)])
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Average Motor Speed [rad/s]')
    ax.legend()
    ax.grid(True)

def plot_rotor_efficiency_comparison(sim_results, controller_types):
    """Plot rotor efficiency comparison"""
    num_controllers = len(controller_types)
    fig = plt.figure(figsize=(8,4*num_controllers))
    gs = fig.add_gridspec(2*num_controllers, 4)

    # Get global min/max values
    pos_error_min, pos_error_max = get_position_error_range(sim_results)
    rotor_speed_min, rotor_speed_max = get_rotor_speed_range(sim_results)
    window = np.ones(20)/20  # 20-point moving average

    for idx, (result, ctrl_type) in enumerate(zip(sim_results, controller_types)):
        ax1 = fig.add_subplot(gs[idx*2:(idx+1)*2, 0:2], projection='3d')
        controller_palette = sns.color_palette("husl", len(controller_types))
        plot_3d_trajectory(ax1, sim_results, controller_types, controller_palette)
        plot_position_errors_subplot(fig, gs[idx*2, 2:], result, ctrl_type, idx==0,
                                   pos_error_min, pos_error_max)
        plot_rotor_speeds_subplot(fig, gs[idx*2+1,2:], result, ctrl_type, idx==0,
                                rotor_speed_min, rotor_speed_max, window)

    plt.tight_layout(h_pad=1.0, w_pad=1.0)
    return fig

def get_position_error_range(sim_results):
    """Calculate global min/max position error"""
    pos_error_min = float('inf')
    pos_error_max = float('-inf')
    for result in sim_results:
        pos_error = result['state']['x'] - result['flat']['x']
        pos_error_min = min(pos_error_min, pos_error.min())
        pos_error_max = max(pos_error_max, pos_error.max())
    return pos_error_min, pos_error_max

def get_rotor_speed_range(sim_results):
    """Calculate global min/max rotor speeds"""
    speed_min = float('inf')
    speed_max = float('-inf')
    for result in sim_results:
        speeds = result['state']['rotor_speeds']
        speed_min = min(speed_min, speeds.min())
        speed_max = max(speed_max, speeds.max())
    return speed_min, speed_max

def plot_position_errors_subplot(fig, pos, result, ctrl_type, is_leftmost, ymin, ymax):
    """Plot position errors for a single controller"""
    ax = fig.add_subplot(pos)
    pos_palette = sns.color_palette("husl", 3)
    time = result['time']
    pos_error = result['state']['x'] - result['flat']['x']
    
    for i, label in enumerate(['X', 'Y', 'Z']):
        ax.plot(time, pos_error[:,i], label=label, color=pos_palette[i])
    
    ax.set_title(f'{ctrl_type}')
    if is_leftmost:
        ax.set_ylabel('Position Error [m]')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(ymin, ymax)

def plot_rotor_speeds_subplot(fig, pos, result, ctrl_type, is_leftmost, ymin, ymax, window):
    """Plot rotor speeds for a single controller"""
    ax = fig.add_subplot(pos)
    rotor_palette = sns.color_palette("husl", 4)
    time = result['time']
    rotor_speeds = result['state']['rotor_speeds']

    window_size = 20
    weights = np.repeat(1.0, window_size)/window_size
    
    for i in range(4):
        smoothed = np.convolve(rotor_speeds[:,i], window, mode='same')
        # Scale the data to motor speed range
        ax.plot(time, smoothed, label=f'Rotor {i+1}', color=rotor_palette[i])
        # ax.plot(time, rotor_speeds[:,i], label=f'Rotor {i+1}', color=rotor_palette[i])
    
    if is_leftmost:
        ax.set_ylabel('Rotor Speed [rad/s]')
    ax.set_xlabel('Time [s]')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(ymin, ymax) 