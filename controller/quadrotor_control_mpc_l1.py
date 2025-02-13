import numpy as np
from time import time
from collections import deque
from scipy.spatial.transform import Rotation
from rotorpy.trajectories.hover_traj  import HoverTraj
from rotorpy.controllers.quadrotor_mpc_1 import QuadMPC
from rotorpy.controllers.quadrotor_util import skew_symmetric, v_dot_q, quaternion_inverse
from rotorpy.learning.util import compute_res
import numpy.linalg as la
import math


class L1_ModelPredictiveControl(object):
    """

    """
    def __init__(self, quad_params, sim_rate, 
                 trajectory, t_final, t_horizon, n_nodes 
                 ):
        """
        Parameters:
            quad_params, dict with keys specified in rotorpy/vehicles
        """
        self.quad_mpc = QuadMPC(quad_params=quad_params, trajectory=trajectory, t_final=t_final,
                                t_horizon=t_horizon, n_nodes=n_nodes)

        # compute optimation rate
        self.optimization_dt = t_horizon / n_nodes
        self.sim_dt = 1/sim_rate
        self.sliding_index = 0 #determine current MPC reference

        # Initilize controls
        self.cmd_motor_thrusts = np.zeros((4,))

        # Quadrotor physical parameters.
        # Inertial parameters
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.Ixy             = quad_params['Ixy']  # kg*m^2
        self.Ixz             = quad_params['Ixz']  # kg*m^2
        self.Iyz             = quad_params['Iyz']  # kg*m^2

        # Frame parameters
        self.c_Dx            = quad_params['c_Dx']  # drag coeff, N/(m/s)**2
        self.c_Dy            = quad_params['c_Dy']  # drag coeff, N/(m/s)**2
        self.c_Dz            = quad_params['c_Dz']  # drag coeff, N/(m/s)**2

        self.num_rotors      = quad_params['num_rotors']
        self.rotor_pos       = quad_params['rotor_pos']
        self.rotor_dir       = quad_params['rotor_directions']

        # Rotor parameters    
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s

        self.k_eta           = quad_params['k_eta']     # thrust coeff, N/(rad/s)**2
        self.k_m             = quad_params['k_m']       # yaw moment coeff, Nm/(rad/s)**2
        self.k_d             = quad_params['k_d']       # rotor drag coeff, N/(m/s)
        self.k_z             = quad_params['k_z']       # induced inflow coeff N/(m/s)
        self.k_flap          = quad_params['k_flap']    # Flapping moment coefficient Nm/(m/s)

        # Motor parameters
        self.tau_m           = quad_params['tau_m']     # motor reponse time, seconds

        # You may define any additional constants you like including control gains.
        self.inertia = np.array([[self.Ixx, self.Ixy, self.Ixz],
                                 [self.Ixy, self.Iyy, self.Iyz],
                                 [self.Ixz, self.Iyz, self.Izz]]) # kg*m^2
        self.J = self.inertia # inertial matrix
        self.g = 9.81 # m/s^2
        
        # Linear map from individual rotor forces to scalar thrust and vector
        # moment applied to the vehicle.
        k = self.k_m/self.k_eta  # Ratio of torque to thrust coefficient. 

        # Below is an automated generation of the control allocator matrix. It assumes that all thrust vectors are aligned
        # with the z axis and that the "sign" of each rotor yaw moment alternates starting with positive for r1. 'TM' = "thrust and moments"
        self.f_to_TM = np.vstack((np.ones((1,self.num_rotors)),
                                  np.hstack([np.cross(self.rotor_pos[key],np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos]), 
                                 (k * self.rotor_dir).reshape(1,-1)))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)

        """ L1-related parameters """
        self.As_v = -1 # parameter for L1
        self.As_omega = -1 # parameter for L1
        self.dt_L1 = 1/100 # sample time for L1 AC, for simplicity, set the same as the simulation step size

        """ For large uncertainties ..."""
        self.ctoffq1Thrust = 50 # cutoff frequency for thrust channel LPF (rad/s)
        self.ctoffq1Moment = 50 # cutoff frequency for moment channels LPF1 (rad/s)
        self.ctoffq2Moment = 50 # cutoff frequency for moment channels LPF2 (rad/s)

        self.L1_params = (self.As_v, self.As_omega, self.dt_L1, self.ctoffq1Thrust, self.ctoffq1Moment, self.ctoffq2Moment, self.mass, self.g, self.J )

        # self.kx = 16*self.m*np.ones((3,)) # position gains
        # self.kv = 5.6*self.m*np.ones((3,)) # velocity gains
        # self.kR = 8.81*np.ones((3,)) # angular gains
        # self.kW = 2.54*np.ones((3,)) # rotational velocity gains

        """ Initialization of L1 inputs """
        self.v_hat_prev = np.array([0.0, 0.0, 0.0])
        self.omega_hat_prev = np.array([0.0, 0.0, 0.0])
        self.R_prev = np.eye(3) 
        self.v_prev = np.array([0.0,0.0,0.0])
        self.omega_prev = np.array([0.0,0.0,0.0])

        self.u_b_prev = np.array([0.0,0.0,0.0,0.0])
        self.u_ad_prev = np.array([0.0,0.0,0.0,0.0])
        self.sigma_m_hat_prev = np.array([0.0,0.0,0.0,0.0])
        self.sigma_um_hat_prev = np.array([0.0,0.0])
        self.lpf1_prev = np.array([0.0,0.0,0.0,0.0])
        self.lpf2_prev = np.array([0.0,0.0,0.0,0.0])
        self.din_L1 = (self.v_hat_prev, self.omega_hat_prev, self.R_prev, self.v_prev, self.omega_prev,
                       self.u_b_prev, self.u_ad_prev, self.sigma_m_hat_prev, self.sigma_um_hat_prev, 
                       self.lpf1_prev, self.lpf2_prev)

    def L1AC(self, R, W, x, v, f, M):
            (As_v, As_omega, dt, ctoffq1Thrust, ctoffq1Moment, ctoffq2Moment, kg_vehicleMass, GRAVITY_MAGNITUDE, J ) = self.L1_params
            (v_hat_prev, omega_hat_prev, R_prev, v_prev, omega_prev,
            u_b_prev, u_ad_prev, sigma_m_hat_prev, sigma_um_hat_prev,
            lpf1_prev, lpf2_prev) = self.din_L1

            # == begin L1 adaptive control ==
            # first do the state predictor
            e3 = np.array([0.0, 0.0, 1.0])
            # load translational velocity
            v_now = v

            # load rotational velocity
            omega_now = W

            massInverse = 1.0 / kg_vehicleMass

            # compute prediction error (on previous step)
            vpred_error_prev = v_hat_prev - v_prev # computes v_tilde for (k-1) step
            omegapred_error_prev = omega_hat_prev - omega_prev # computes omega_tilde for (k-1) step

            v_hat = v_hat_prev + (-e3 * GRAVITY_MAGNITUDE + R_prev[:,2]* (u_b_prev[0] + u_ad_prev[0] + sigma_m_hat_prev[0]) * massInverse + R_prev[:,0] * sigma_um_hat_prev[0] * massInverse + R_prev[:,1] * sigma_um_hat_prev[1] * massInverse + vpred_error_prev * As_v) * dt
           
            Jinv = la.inv(J)
            # temp vector: thrustMomentCmd[1--3] + u_ad_prev[1--3] + sigma_m_hat_prev[1--3]
            # original form
            tempVec = np.array([u_b_prev[1] + u_ad_prev[1] + sigma_m_hat_prev[1], u_b_prev[2] + u_ad_prev[2] + sigma_m_hat_prev[2], u_b_prev[3] + u_ad_prev[3] + sigma_m_hat_prev[3]])
            omega_hat = omega_hat_prev + (-np.matmul(Jinv, np.cross(omega_prev, (np.matmul(J, omega_prev)))) + np.matmul(Jinv, tempVec) + omegapred_error_prev * As_omega) * dt

            # update the state prediction storage
            v_hat_prev = v_hat
            omega_hat_prev = omega_hat

            # compute prediction error (for this step)
            vpred_error = v_hat - v_now
            omegapred_error = omega_hat - omega_now

            # exponential coefficients coefficient for As
            exp_As_v_dt = math.exp(As_v * dt)
            exp_As_omega_dt = math.exp(As_omega * dt)

            # latter part of uncertainty estimation (piecewise constant) (step2: adaptation law)
            PhiInvmu_v = vpred_error / (exp_As_v_dt - 1) * As_v * exp_As_v_dt
            PhiInvmu_omega = omegapred_error / (exp_As_omega_dt - 1) * As_omega * exp_As_omega_dt

            sigma_m_hat = np.array([0.0,0.0,0.0,0.0]) # estimated matched uncertainty
            sigma_m_hat_2to4 = np.array([0.0,0.0,0.0]) # second to fourth element of the estimated matched uncertainty
            sigma_um_hat = np.array([0.0,0.0]) # estimated unmatched uncertainty

            sigma_m_hat[0] = -np.dot(R[:,2], PhiInvmu_v) * kg_vehicleMass # don't forget the minus
            # turn np.dot(R[:,2], PhiInvmu_v) * kg_vehicleMass to -np.dot(R[:,2], PhiInvmu_v) * kg_vehicleMass
            sigma_m_hat_2to4 = -np.matmul(J, PhiInvmu_omega)
            sigma_m_hat[1] = sigma_m_hat_2to4[0]
            sigma_m_hat[2] = sigma_m_hat_2to4[1]
            sigma_m_hat[3] = sigma_m_hat_2to4[2]

            sigma_um_hat[0] = -np.dot(R[:,0], PhiInvmu_v) * kg_vehicleMass
            sigma_um_hat[1] = -np.dot(R[:,1], PhiInvmu_v) * kg_vehicleMass

            # store uncertainty estimations
            sigma_m_hat_prev = sigma_m_hat
            sigma_um_hat_prev = sigma_um_hat

            # compute lpf1 coefficients
            lpf1_coefficientThrust1 = math.exp(- ctoffq1Thrust * dt)
            lpf1_coefficientThrust2 = 1.0 - lpf1_coefficientThrust1

            lpf1_coefficientMoment1 = math.exp(- ctoffq1Moment * dt)
            lpf1_coefficientMoment2 = 1.0 - lpf1_coefficientMoment1

            # update the adaptive control
            u_ad_int = np.array([0.0,0.0,0.0,0.0])
            u_ad = np.array([0.0,0.0,0.0,0.0])

            # low-pass filter 1 (negation is added to u_ad_prev to filter the correct signal)
            u_ad_int[0] = lpf1_coefficientThrust1 * (lpf1_prev[0]) + lpf1_coefficientThrust2 * sigma_m_hat[0]
            u_ad_int[1:4] = lpf1_coefficientMoment1 * (lpf1_prev[1:4]) + lpf1_coefficientMoment2 * sigma_m_hat[1:4]

            lpf1_prev = u_ad_int # store the current state

            # coefficients for the second LPF on the moment channel
            lpf2_coefficientMoment1 = math.exp(- ctoffq2Moment * dt)
            lpf2_coefficientMoment2 = 1.0 - lpf2_coefficientMoment1

            # low-pass filter 2 (optional)
            u_ad[0] = u_ad_int[0] # only one filter on the thrust channel
            u_ad[1:4] = lpf2_coefficientMoment1 * lpf2_prev[1:4] + lpf2_coefficientMoment2 * u_ad_int[1:4]

            lpf2_prev = u_ad # store the current state

            u_ad = -u_ad

            # store the values for next iteration (negation is added to u_ad_prev to filter the correct signal)
            u_ad_prev = u_ad

            v_prev = v_now
            omega_prev = omega_now
            R_prev = R
            u_b_prev = np.array([f,M[0],M[1],M[2]])

            controlcmd_L1 = np.array([f,M[0],M[1],M[2]]) + u_ad_prev

            self.din_L1 = (v_hat_prev, omega_hat_prev, R_prev, v_prev, omega_prev,
            u_b_prev, u_ad_prev, sigma_m_hat_prev, sigma_um_hat_prev,
            lpf1_prev, lpf2_prev)

            f_L1 = controlcmd_L1[0]
            M_L1 = controlcmd_L1[1:4]
            return (f_L1, M_L1, sigma_m_hat)


    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N 
                cmd_moment, N*m
                cmd_q, quaternion [i,j,k,w]
        """
        
        x = state['x'].reshape(3)
        v = state['v'].reshape(3)
        R = Rotation.from_quat(state['q']).as_matrix()     
        W = state['w'].reshape(3)

        # unpack state used for MPC
        task_index = None
        state = self.unpack_state(state)


        # Optimization loop
        index, _ = divmod(t, self.optimization_dt)
        if int(index) == self.sliding_index:
            self.quad_mpc.set_reference(self.sliding_index)
            w_opt,x_opt,sens_u = self.quad_mpc.run_optimization(initial_state=state, task_index=task_index)
            self.cmd_motor_thrusts = w_opt[:4]   # get controls
            # cmd_motor_thrusts = self.cmd_motor_thrusts
            # cmd_motor_speeds = cmd_motor_thrusts / self.k_eta
            # cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))
            self.sliding_index += 1             # update slidng index

        # Compute motor speeds. Avoid taking square root of negative numbers.

        TM = self.f_to_TM @ self.cmd_motor_thrusts
        f = TM[0]
        M = TM[1:4]
        f_l1, M_l1, sigma_m_hat = self.L1AC(R,W,x,v,f,M)

        u_new = np.vstack((f_l1.reshape(1,1),M_l1.reshape(3,1)))

        cmd_TM = np.array(u_new)
        cmd_motor_thrusts = self.TM_to_f @ cmd_TM
        cmd_motor_speeds = cmd_motor_thrusts / self.k_eta
        cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

        cmd_thrust = cmd_TM[0]
        cmd_moment = np.array([cmd_TM[1], cmd_TM[2], cmd_TM[3]])
        cmd_q = np.zeros((4,)) # 
        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}  # This dict is required by simulation env
        

        return control_input
    
    def unpack_state(self, state):
        """
        This function unpacks the state and returns an array [x, v, quaternion(wxyz), w] of shape (13,)
        """
        x = state['x']
        v = state['v']
        q_ = state['q']
        w = state['w']
        
        # Note: MPC uses quaternion as wxyz instead of xyzw used by the SIMULATOR
        q = np.array([q_[3], q_[0], q_[1], q_[2]])
        return np.concatenate([x,v,q,w])
    