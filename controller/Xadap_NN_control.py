"""
Python implementation of Learning-based Controller for Extreme Adaptation based on:
Dingqi Zhang, Antonio Loquercio, Jerry Tang, Ting-Hao Wang, Jitendra Malik, Mark W. Mueller.
"A Learning-based Quadcopter Controller with Extreme Adaptation"
[https://arxiv.org/abs/2409.12949]
"""

import numpy as np
from scipy.spatial.transform import Rotation
from controller.controller_template import MultirotorControlTemplate
from controller.geometric_control import GeoControl
from rotorpy.controllers.quadrotor_control import SE3Control

import numpy as np
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
from enum import Enum
import os
from collections import deque
import pandas as pd


class ModelType(Enum):
    BASE_MODEL = 1
    ADAP_MODULE = 2
class Model:
    def __init__(self, base_model_path='mlp_1011.onnx', adap_module_path='encoder_1011.onnx', model_rms_path=None):
        current_path = os.path.dirname(os.path.abspath(__file__)) + '/models/Xadap/'
        self.base_model_path = current_path+base_model_path
        self.adap_module_path = current_path+adap_module_path
        # self.model_rms_path = current_path+model_rms_path

        self.obs_mean = np.loadtxt(current_path+'mean_1011.csv', delimiter=" ")
        self.obs_var = np.loadtxt(current_path+'var_1011.csv', delimiter=" ")


        self.act_mean = np.array([1.0 / 2, 1.0 / 2,
                                  1.0 / 2, 1.0 / 2])[np.newaxis, :]
        self.act_std = np.array([1.0 / 2, 1.0 / 2,
                                 1.0 / 2, 1.0 / 2])[np.newaxis, :]
        self.act_size = 4
        self.state_obs_size = 8
        self.history_len = 100

        self.base_session = None
        self.adap_session = None
        self.base_obs_name = None
        self.adap_obs_name = None

    def set_act_size(self, act_size):
        self.act_size = act_size

    def set_state_obs_size(self, state_obs_size):
        self.state_obs_size = state_obs_size

    def set_history_len(self, history_len):
        self.history_len = history_len
        
    def set_const_sizes(self,state_obs_size,act_size,history_len):
        self.set_act_size(act_size)
        self.set_state_obs_size(state_obs_size)
        self.set_history_len(history_len)

    def activate(self):
        self.base_session = onnxruntime.InferenceSession(
            self.base_model_path, None)
        self.base_obs_name = self.base_session.get_inputs()[0].name
        self.adap_session = onnxruntime.InferenceSession(
            self.adap_module_path, None)
        self.adap_obs_name = self.adap_session.get_inputs()[0].name

    def normalize_obs(self, obs, model_type):
        if model_type is ModelType.BASE_MODEL:
            return (obs - self.obs_mean[:self.state_obs_size+self.act_size]) / np.sqrt(self.obs_var[:self.state_obs_size+self.act_size] + 1e-8)
        else:
            # Normalize for Adaptation module observations
            obs_n_norm = obs.reshape([1, -1])

            # state normalization
            obs_state_history_n_normalized = obs_n_norm[:, -self.history_len*(
                self.act_size+self.state_obs_size):-self.history_len*self.act_size]

            obs_state_mean = np.tile(self.obs_mean[:self.state_obs_size], [
                                     1, self.history_len])
            obs_state_var = np.tile(self.obs_var[:self.state_obs_size], [
                                    1, self.history_len])

            obs_state_history_normalized = (
                obs_state_history_n_normalized - obs_state_mean) / np.sqrt(obs_state_var + 1e-8)

            obs_n_norm[:, -self.history_len*(self.act_size+self.state_obs_size):-self.history_len*self.act_size] = obs_state_history_normalized

            # action normalization
            obs_act_mean = np.tile(self.obs_mean[self.state_obs_size:self.state_obs_size+self.act_size], [
                                      1, self.history_len])
            obs_act_var = np.tile(self.obs_var[self.state_obs_size:self.state_obs_size+self.act_size], [
                                        1, self.history_len])
            
            obs_act_history_n_normalized = obs_n_norm[:, -self.history_len*self.act_size:]
            obs_act_history_normalized = (
                obs_act_history_n_normalized - obs_act_mean) / np.sqrt(obs_act_var + 1e-8)
            obs_n_norm[:, -self.history_len*self.act_size:] = obs_act_history_normalized

            obs_norm = obs_n_norm

            return obs_norm

    def run(self, cur_obs, last_act, obs_history, act_history):
        norm_cur_obs = self.normalize_obs(np.concatenate(
            (cur_obs, last_act)), model_type=ModelType.BASE_MODEL)
        norm_history = self.normalize_obs(np.concatenate(
            (obs_history, act_history)), model_type=ModelType.ADAP_MODULE)
        latent = self.adap_session.run(
            None, {self.adap_obs_name: norm_history})
        obs = np.concatenate((norm_cur_obs.reshape(1,-1), np.asarray(latent).reshape((1,-1))),axis=1).astype(np.float32)
        raw_act = np.asarray(self.base_session.run(None, {self.base_obs_name: obs})).squeeze()
        norm_action = (raw_act * self.act_std + self.act_mean)[0, :]
        return norm_action, raw_act




    
# inputs current raw data (human-readable, non-normalized), outputs actual motor speed
class XAdapLowLevelControl:
    def __init__(self):

        # time
        self.t = 0
        # Learning-based controller
        self.model = Model()

        
        self.maxMotorSpd = 5000 
        
        self.state_vars = ['wx', 'wy', 'wz', 'prop_acc', 'cmd_wx', 'cmd_wy', 'cmd_wz', 'cmd_prop_acc']
        self.action_vars = ['act1', 'act2', 'act3', 'act4']
        
        history_len = 100
        act_size = len(self.action_vars)
        state_obs_size = len(self.state_vars)
        
        self.cur_obs = np.zeros((state_obs_size,))
        self.last_act = np.zeros((act_size,))
        
        
        self.model.set_const_sizes(state_obs_size,act_size,history_len)
        
        self.obs_history = deque([np.zeros(state_obs_size)]*history_len)

        self.act_history = deque([np.zeros(act_size)]*history_len)
        
        self.model.activate()
        
    def set_max_motor_spd(self,max_motor_spd):
        self.maxMotorSpd = max_motor_spd
        
        
    def run(self,cur_obs):
        
        norm_act, raw_act = self.model.run(cur_obs,
            self.last_act,np.asarray(self.obs_history, dtype=np.float32).flatten(),
            np.asarray(self.act_history, dtype=np.float32).flatten())
        
        
        self.obs_history.popleft()
        self.obs_history.append(cur_obs)
        self.act_history.popleft()
        self.act_history.append(raw_act)
        
        self.last_act = raw_act
        
        # Drone model for NN is    
        #           
        #           x
        #           ^
        #      mot3+| mot0-
        #           |
        #     y<----+-----
        #           |
        #      mot1-| mot2+
        #    
        spd_NN = norm_act.squeeze() * self.maxMotorSpd
        # hardcode to fit simulate drone model

        # Drone model for Simulation is    
        #           
        #           x
        #           ^
        #      mot0+| mot1-
        #           |
        #     y<----+-----
        #           |
        #      mot3-| mot2+
        #    
        
        spd_cmd = np.zeros(4,)
        spd_cmd[0] = spd_NN[3]
        spd_cmd[1] = spd_NN[0]
        spd_cmd[2] = spd_NN[2]
        spd_cmd[3] = spd_NN[1]
        return spd_cmd


class Xadap_NN_control(MultirotorControlTemplate):
    def __init__(self, vehicle_params):
        super().__init__(vehicle_params)
        self.high_level_control = SE3Control(vehicle_params)
        self.low_level_control = XAdapLowLevelControl()
        self.low_level_control.set_max_motor_spd(self.rotor_speed_max)

    def update(self, t, state, flat_output):

        # Import Geometric to get des angular vel & THRUST
        high_level_control_input = self.high_level_control.update(t, state, flat_output)
        cmd_w = high_level_control_input['cmd_w']
        cmd_thrust = high_level_control_input['cmd_thrust']
        cmd_mass_norm_thrust = cmd_thrust / self.mass
        cmd_mass_norm_thrust = np.minimum(np.maximum(cmd_mass_norm_thrust,0.01),20.0)
        cmd_thrust = cmd_mass_norm_thrust * self.mass


        # Estimated acceleration in body frame
        R = Rotation.from_quat(state['q']).as_matrix()
        meas_acc = state['accel']
        

        rotation_matrix = R.reshape((9,), order="F")
        cur_obs = np.concatenate((
            # rotation_matrix, 
                                          state['w'], 
                                          np.array([meas_acc[-1]],dtype=np.float32),  
                                          cmd_w.reshape(-1,),
                                          np.array([cmd_mass_norm_thrust],dtype=np.float32),  
                                                    ), axis=0).astype(np.float32)
        
        
        cmd_motor_speeds = self.low_level_control.run(cur_obs)
        cmd_motor_thrusts = np.zeros(4,)
        cmd_moment = high_level_control_input['cmd_moment']
        cmd_q = high_level_control_input['cmd_q']
        cmd_v = high_level_control_input['cmd_v']

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_motor_thrusts':cmd_motor_thrusts,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q,
                         'cmd_w':cmd_w,
                         'cmd_v':cmd_v}
        
        return control_input



