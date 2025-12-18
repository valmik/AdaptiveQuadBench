"""
OmniDrone to AdaptiveQuadBench Bridge Controller

This controller bridges OmniDrone-trained models to work with AdaptiveQuadBench simulator.
It handles conversion between observation formats, applies transforms manually, and converts
normalized motor commands to actual motor speeds.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from aqb.controller.controller_template import MultirotorControlTemplate
from tensordict import TensorDict


class OmniDroneBridge(MultirotorControlTemplate):
    """
    Bridge controller that runs OmniDrone-trained models in AdaptiveQuadBench.
    
    Takes pre-initialized policy (neural network) and optional transform as constructor arguments.
    Handles conversion between AQB state format and OmniDrone observation format, applies
    transforms manually, and converts normalized motor commands to motor speeds.
    """
    
    def __init__(self, vehicle_params, policy, transform=None, device='cpu', cfg=None):
        """
        Initialize the bridge controller.
        
        Parameters:
            vehicle_params: dict with vehicle parameters (as required by MultirotorControlTemplate)
            policy: Pre-loaded and initialized OmniDrone policy (e.g., PPOPolicy)
            transform: Optional pre-initialized controller transform (e.g., HybridPositionControllerVelocity)
                      If None, assumes end-to-end control (network outputs motor commands directly)
            device: torch device for computations ('cpu' or 'cuda')
            cfg: OmniDrone config object (OmegaConf) containing task configuration.
                 If provided, extracts future_traj_steps and time_encoding from cfg.task.
                 If None, uses defaults (future_traj_steps=4, time_encoding=False).
        """
        super().__init__(vehicle_params)
        
        self.policy = policy
        self.transform = transform
        self.device = torch.device(device)
        
        # Extract parameters from config if provided
        if cfg is not None:
            self.future_traj_steps = cfg.task.get('future_traj_steps', 4)
            self.time_encoding = cfg.task.get('time_encoding', False)
        else:
            self.future_traj_steps = 4
            self.time_encoding = False
        
        self.time_encoding_dim = 4 if self.time_encoding else 0
        
        # Set policy to eval mode
        self.policy.eval()
        
        # Move policy to device if needed
        if hasattr(self.policy, 'to'):
            self.policy = self.policy.to(self.device)
        
        # Move transform to device if needed
        if self.transform is not None and hasattr(self.transform, 'to'):
            self.transform = self.transform.to(self.device)
    
    # TODO: We need to pass in the full trajectory, not just the current output. There are MPC implementations, so some of this infrastructure may be reusable.
    # This is because the integration is going to be very lossy
    # Also the time encoding needs a max time
    # TODO: We may need to pass in the OmniDrone environment so we can get dt
    def _aqb_state_to_omni_obs(self, t, state, flat_output):
        """
        Convert AdaptiveQuadBench state format to OmniDrone observation format.
        
        Parameters:
            t: current time
            state: AQB state dict with keys x, v, q, w
            flat_output: AQB flat_output dict with trajectory information
            
        Returns:
            obs: torch.Tensor of shape (1, obs_dim) matching OmniDrone format
            drone_state: torch.Tensor of shape (19 + num_rotors,) for transform
            future_targets: torch.Tensor of shape (future_traj_steps, 3) for transform
            future_times: torch.Tensor of shape (future_traj_steps,) for transform
        """
        # --- Work in numpy on CPU for SciPy + linear algebra, then convert to torch on self.device ---
        # Ensure we have numpy arrays (handles both np.array and torch.Tensor inputs gracefully)
        pos_np = np.asarray(state['x'], dtype=np.float32)
        vel_world_np = np.asarray(state['v'], dtype=np.float32)   # world-frame linear velocity
        q_np = np.asarray(state['q'], dtype=np.float32)           # quaternion [x, y, z, w]
        w_body_np = np.asarray(state['w'], dtype=np.float32)      # body-frame angular velocity

        # Compute heading and up vectors from quaternion using SciPy (expects numpy on CPU)
        rot = Rotation.from_quat(q_np)  # [x, y, z, w]
        rot_matrix = rot.as_matrix()    # 3x3 rotation matrix
        heading_np = rot_matrix[:, 0]   # body x-axis in world frame
        up_np = rot_matrix[:, 2]        # body z-axis in world frame

        # Convert world-frame linear velocity to body frame: v_body = R^T * v_world
        vel_body_np = rot_matrix.T @ vel_world_np

        # Now move everything to torch tensors on the target device
        pos = torch.from_numpy(pos_np).to(self.device)
        vel_body = torch.from_numpy(vel_body_np).to(self.device)
        w_body = torch.from_numpy(w_body_np).to(self.device)
        q = torch.from_numpy(q_np).to(self.device)
        heading = torch.from_numpy(heading_np).to(self.device)
        up = torch.from_numpy(up_np).to(self.device)
        
        # Compute future trajectory positions from flat_output
        # Use derivatives to estimate future positions
        current_pos = torch.tensor(flat_output['x'], dtype=torch.float32, device=self.device)
        current_vel = torch.tensor(flat_output['x_dot'], dtype=torch.float32, device=self.device)
        current_acc = torch.tensor(flat_output['x_ddot'], dtype=torch.float32, device=self.device)
        
        # Estimate future positions using constant acceleration model
        # For simplicity, use dt = 0.1s per step (can be made configurable)
        dt = 0.1
        future_targets = []
        future_times = []
        
        for i in range(self.future_traj_steps):
            t_future = t + (i + 1) * dt
            # Simple integration: x = x0 + v0*t + 0.5*a*t^2
            pos_future = current_pos + current_vel * (i + 1) * dt + 0.5 * current_acc * ((i + 1) * dt) ** 2
            future_targets.append(pos_future)
            future_times.append(t_future)
        
        future_targets = torch.stack(future_targets, dim=0)  # (future_traj_steps, 3)
        future_times = torch.tensor(future_times, dtype=torch.float32, device=self.device)  # (future_traj_steps,)
        
        # Compute relative positions (rpos)
        rpos = future_targets - pos.unsqueeze(0)  # (future_traj_steps, 3)
        
        # Construct drone state (excluding position, starting from rotation)
        # Format: [rot(4), vel(6), heading(3), up(3), throttle(num_rotors)]
        # Based on multirotor.py get_state(), vel is 6D: [vx_b, vy_b, vz_b, wx_b, wy_b, wz_b]
        # where velocities are in body frame
        vel_combined = torch.cat([vel_body, w_body], dim=0)  # (6,)
        
        # For throttle, we don't have it from AQB, so use zeros (normalized)
        throttle = torch.zeros(self.num_rotors, dtype=torch.float32, device=self.device)
        
        drone_state = torch.cat([
            q,           # (4,) quaternion
            vel_combined,  # (6,) velocities
            heading,     # (3,) heading
            up,          # (3,) up
            throttle     # (num_rotors,) throttle
        ], dim=0)  # Total: 4 + 6 + 3 + 3 + num_rotors = 16 + num_rotors
        
        # Add position at the beginning for full state (needed for some transforms)
        drone_state_full = torch.cat([pos, drone_state], dim=0)  # (19 + num_rotors,)
        
        # Construct observation: [rpos_flat, drone_state[3:], (optional time_encoding)]
        # Based on track_simpleflight.py, rpos is flattened from (future_traj_steps, 3)
        rpos_flat = rpos.flatten()  # (3 * future_traj_steps,)
        
        obs_parts = [
            rpos_flat,                    # (3 * future_traj_steps,)
            drone_state,                  # (16 + num_rotors,)
        ]
        
        if self.time_encoding:
            # Simple time encoding: [sin(2πt/T), cos(2πt/T), sin(4πt/T), cos(4πt/T)]
            # Using a default episode length T=30s
            T = 30.0
            time_enc = torch.tensor([
                np.sin(2 * np.pi * t / T),
                np.cos(2 * np.pi * t / T),
                np.sin(4 * np.pi * t / T),
                np.cos(4 * np.pi * t / T)
            ], dtype=torch.float32, device=self.device)
            obs_parts.append(time_enc)
        
        obs = torch.cat(obs_parts, dim=0)  # (obs_dim,)
        # Ensure observation is float32 to match PPO network weights
        obs = obs.to(dtype=torch.float32)
        obs = obs.unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim) to match OmniDrone format
        
        return obs, drone_state_full, future_targets.unsqueeze(0), future_times.unsqueeze(0)
    
    def _apply_transform(self, obs, drone_state, future_targets, future_times):
        """
        Apply OmniDrone transform to process network actions.
        
        Parameters:
            obs: observation tensor
            drone_state: full drone state tensor
            future_targets: future trajectory positions
            future_times: future trajectory times
            
        Returns:
            normalized_motor_cmds: torch.Tensor of shape (num_rotors,) with normalized commands [-1, 1]
        """
        if self.transform is None:
            # End-to-end control: network directly outputs motor commands
            # Create a simple tensordict for the policy
            tensordict = TensorDict({
                ("agents", "observation"): obs,
            }, batch_size=[1], device=self.device)
            
            # Get action from policy (policies modify TensorDict in place)
            with torch.no_grad():
                tensordict = self.policy(tensordict)
                action = tensordict[("agents", "action")]
            
            # Action should already be normalized motor commands
            # Reshape if needed - remove batch and agent dimensions
            while action.dim() > 1:
                action = action.squeeze(0)
            
            if action.shape[-1] == self.num_rotors:
                return action
            else:
                raise ValueError(f"Expected action shape with {self.num_rotors} motors, got {action.shape}")
        
        # With transform: need to construct proper TensorDict
        tensordict = TensorDict({
            ("agents", "observation"): obs,
            ("info", "drone_state"): drone_state.unsqueeze(0),  # Add batch dimension
            ("traj_stats", "future_targets"): future_targets,  # (1, future_traj_steps, 3)
            ("traj_stats", "future_times"): future_times,  # (1, future_traj_steps)
        }, batch_size=[1], device=self.device)
        
        # Get action from policy (policies modify TensorDict in place)
        with torch.no_grad():
            tensordict = self.policy(tensordict)
            action = tensordict[("agents", "action")]
        
        # Apply transform
        tensordict = self.transform._inv_call(tensordict)
        
        # Extract motor commands from transform output
        motor_cmds = tensordict[("agents", "action")]
        
        # Reshape to (num_rotors,) - remove batch and agent dimensions
        while motor_cmds.dim() > 1:
            motor_cmds = motor_cmds.squeeze(0)
        
        if motor_cmds.shape[-1] == self.num_rotors:
            return motor_cmds
        else:
            raise ValueError(f"Expected motor commands shape with {self.num_rotors} motors, got {motor_cmds.shape}")
    
    def _normalized_to_motor_speeds(self, normalized_cmds):
        """
        Convert normalized motor commands [-1, 1] to motor speeds in rad/s.
        
        Parameters:
            normalized_cmds: torch.Tensor of shape (num_rotors,) with values in [-1, 1]
            
        Returns:
            motor_speeds: numpy array of shape (num_rotors,) with values in [0, rotor_speed_max]
        """
        # Convert to numpy
        if isinstance(normalized_cmds, torch.Tensor):
            normalized_cmds = normalized_cmds.cpu().numpy()
        
        # Map from [-1, 1] to [0, rotor_speed_max]
        # Formula: motor_speed = (normalized_cmd + 1) / 2 * rotor_speed_max
        motor_speeds = (normalized_cmds + 1) / 2 * self.rotor_speed_max
        
        # Clip to valid range
        motor_speeds = np.clip(motor_speeds, 0, self.rotor_speed_max)
        
        return motor_speeds
    
    def update(self, t, state, flat_output):
        """
        Main update function called by AdaptiveQuadBench simulator.
        
        Parameters:
            t: current time in seconds
            state: dict with keys x, v, q, w
            flat_output: dict with trajectory information
            
        Returns:
            control_input: dict with cmd_motor_speeds key
        """
        # Convert AQB state to OmniDrone observation format
        obs, drone_state, future_targets, future_times = self._aqb_state_to_omni_obs(t, state, flat_output)
        
        # Apply transform and get normalized motor commands
        normalized_cmds = self._apply_transform(obs, drone_state, future_targets, future_times)
        
        # Convert normalized commands to motor speeds
        motor_speeds = self._normalized_to_motor_speeds(normalized_cmds)
        
        # Build full control dict expected by RotorPy/AQB
        # Many of these fields are not used by the simulator for dynamics,
        # but are expected to exist for cost computation and plotting.
        cmd_q = state.get('q', np.array([0.0, 0.0, 0.0, 1.0]))
        cmd_w = state.get('w', np.zeros(3,))
        cmd_moment = np.zeros(3,)
        cmd_thrust = 0.0
        
        # Return control input in AQB format
        control_input = {
            'cmd_motor_speeds': motor_speeds,
            'cmd_q': cmd_q,
            'cmd_w': cmd_w,
            'cmd_moment': cmd_moment,
            'cmd_thrust': cmd_thrust,
        }
        
        return control_input

