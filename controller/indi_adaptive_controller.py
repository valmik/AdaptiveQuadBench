"""
Python implementation of INDI Adaptive Controller based on:
Ewoud J. J. Smeur, Qiping Chu and Guido C. H. E. de Croon
"Adaptive incremental nonlinear dynamic inversion for attitude control of micro air vehicles"
[DOI:10.2514/1.G001490]
"""

import numpy as np
from scipy.spatial.transform import Rotation
from controller.controller_template import MultirotorControlTemplate
from controller.quadrotor_util import *
from controller.geometric_control import GeoControl
from rotorpy.controllers.quadrotor_control import SE3Control

class INDIAdaptiveController(MultirotorControlTemplate):
    def __init__(self, vehicle_params, dt=0.01):
        super().__init__(vehicle_params)
        self.dt = dt
        # Control Effectiveness Matrix
        # # * INDIA paper version
        # self.G1 = 2 * self.k_eta * np.linalg.inv(self.inertia) @ self.f_to_TM[1:,:]
        # self.G2 = np.zeros((3,4))
        
        # * Modified version
        self.G1 = np.ones((4,1))* 0.25 * self.mass / self.k_eta
        self.G2 = self.TM_to_f[:,1:] @ self.inertia / self.k_eta

        # # Adaptive Gain
        self.mu = 1e-7

        # High-level Control
        self.high_level_control = SE3Control(vehicle_params)
        # self.high_level_control = GeoControl(vehicle_params)

        self.last_cmd_motor_speeds = np.zeros((4,1))
        self.last_meas_motor_speeds = np.zeros((4,1))
        self.last_meas_motor_speeds_dot = np.zeros((4,1))
        self.last_meas_omega = np.zeros((3,1))
        self.last_meas_alpha = np.zeros((3,1))

        self.delta_alpha = np.zeros((3,1))
        self.delta_thrust = np.zeros((1,1))


    def update(self, t, state, flat_output):
        # Estimated motor speed 
        meas_motor_speeds = state['rotor_speeds'].reshape(-1,1)
        meas_motor_speeds_dot = (meas_motor_speeds - self.last_meas_motor_speeds) / self.dt

        # Estimated alpha 
        meas_alpha = ((state['w'].reshape(-1,1) - self.last_meas_omega) / self.dt)
        # Estimated acceleration in body frame
        meas_acc = state['accel']

        # Import Geometric to get des Tau & THRUST
        high_level_control_input = self.high_level_control.update(t, state, flat_output)
        cmd_moment = high_level_control_input['cmd_moment']
        cmd_alpha = np.linalg.inv(self.inertia) @ (cmd_moment - np.cross(state['w'], self.inertia @ state['w']))
        cmd_alpha = cmd_alpha.reshape(-1,1)
        cmd_mass_norm_thrust = high_level_control_input['cmd_thrust'] / self.mass

        
        # # # * INDIA paper version
        # G_sum_pinv = np.linalg.pinv(self.G1 + self.G2)
        # cmd_motor_speeds = meas_motor_speeds + G_sum_pinv @ (cmd_alpha - meas_alpha + self.G2 @ (self.last_cmd_motor_speeds - self.last_meas_motor_speeds))
        # cmd_motor_speeds = np.minimum(np.maximum(cmd_motor_speeds, self.rotor_speed_min), self.rotor_speed_max)
        # * Modified version
        mu = self.G1 * (cmd_mass_norm_thrust - np.linalg.norm(meas_acc)) + self.G2 @ (cmd_alpha - meas_alpha)
        cmd_motor_speeds = (meas_motor_speeds ** 2 + mu)
        cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

        cmd_motor_thrusts = self.k_eta * cmd_motor_speeds**2

        # Adaptive Law
        G = np.hstack([self.G1, self.G2])

        # # * INDIA paper version
        # delta_state = np.vstack((meas_motor_speeds - self.last_meas_motor_speeds, 
        #                          meas_motor_speeds_dot - self.last_meas_motor_speeds_dot))
        # G = G - self.mu * (G @ delta_state - (meas_alpha - self.last_meas_alpha)) @ delta_state.T

        # self.G1 = G[:, :4]
        # self.G2 = G[:, 4:]
        # * Modified version
        delta_state = np.vstack((self.delta_thrust, self.delta_alpha))
        G = G - self.mu * (G @ delta_state - (meas_motor_speeds**2 - self.last_meas_motor_speeds**2)) @ delta_state.T

        self.G1 = G[:, :1]
        self.G2 = G[:, 1:]

        self.last_cmd_motor_speeds = cmd_motor_speeds
        self.last_meas_motor_speeds = meas_motor_speeds
        self.last_meas_motor_speeds_dot = meas_motor_speeds_dot
        self.last_meas_alpha = meas_alpha
        self.last_meas_omega = state['w'].reshape(-1,1)
        self.delta_alpha = cmd_alpha - meas_alpha
        self.delta_thrust = cmd_mass_norm_thrust - np.linalg.norm(meas_acc)

        # Only some of these are necessary depending on your desired control abstraction. 
        cmd_thrust = high_level_control_input['cmd_thrust']
        cmd_q = high_level_control_input['cmd_q']
        cmd_w = high_level_control_input['cmd_w']
        cmd_v = high_level_control_input['cmd_v']

        control_input = {'cmd_motor_speeds':cmd_motor_speeds.reshape(-1,),
                         'cmd_motor_thrusts':cmd_motor_thrusts,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q,
                         'cmd_w':cmd_w,
                         'cmd_v':cmd_v}
        
        return control_input