"""
OmniDrone Direct Controller Bridge

This controller bridges OmniDrone's LeePositionController to work with AdaptiveQuadBench simulator.
It handles conversion between AQB state format and the root_state format expected by LeePositionController,
and converts normalized motor commands to actual motor speeds.
"""

import numpy as np
import torch
from aqb.controller.controller_template import MultirotorControlTemplate


class OmniDroneDirect(MultirotorControlTemplate):
    """
    Bridge controller that runs OmniDrone's LeePositionController in AdaptiveQuadBench.
    
    Takes a pre-initialized LeePositionController object as constructor argument.
    Handles conversion between AQB state format and root_state format, extracts targets
    from flat_output, and converts normalized motor commands to motor speeds.
    """
    
    def __init__(self, vehicle_params, controller, device='cpu'):
        """
        Initialize the direct bridge controller.
        
        Parameters:
            vehicle_params: dict with vehicle parameters (as required by MultirotorControlTemplate)
            controller: Pre-initialized LeePositionController object
            device: torch device for computations ('cpu' or 'cuda')
        """
        super().__init__(vehicle_params)
        
        self.controller = controller
        self.device = torch.device(device)

        self.max_thrusts = self.controller.max_thrusts.detach().cpu().numpy() if hasattr(self.controller.max_thrusts, 'detach') else float(self.controller.max_thrusts)

        # Move controller to device if needed
        if hasattr(self.controller, 'to'):
            self.controller = self.controller.to(self.device)
        
        # Set controller to eval mode if applicable
        if hasattr(self.controller, 'eval'):
            self.controller.eval()
    
    def _aqb_state_to_root_state(self, state):
        """
        Convert AdaptiveQuadBench state format to root_state format expected by LeePositionController.
        
        Parameters:
            state: AQB state dict with keys x, v, q, w
            
        Returns:
            root_state: torch.Tensor of shape (13,) with format [pos(3), quat(4), vel_world(3), w_body(3)]
        """
        # Ensure we have numpy arrays (handles both np.array and torch.Tensor inputs gracefully)
        pos_np = np.asarray(state['x'], dtype=np.float32)
        vel_world_np = np.asarray(state['v'], dtype=np.float32)   # world-frame linear velocity
        # Convert input quaternion from [x, y, z, w] (xyzw) to [w, x, y, z] (wxyz) ordering
        q_xyzw = np.asarray(state['q'], dtype=np.float32)
        q_np = np.concatenate([q_xyzw[3:4], q_xyzw[:3]])  # [w, x, y, z]
        w_body_np = np.asarray(state['w'], dtype=np.float32)      # body-frame angular velocity
        
        # Convert to torch tensors on the target device
        pos = torch.from_numpy(pos_np).to(self.device)
        vel_world = torch.from_numpy(vel_world_np).to(self.device)
        q = torch.from_numpy(q_np).to(self.device)
        w_body = torch.from_numpy(w_body_np).to(self.device)
        
        # Construct root_state: [pos(3), quat(4), vel_world(3), w_body(3)]
        root_state = torch.cat([
            pos,        # (3,) position
            q,          # (4,) quaternion [x, y, z, w]
            vel_world,  # (3,) world-frame linear velocity
            w_body      # (3,) body-frame angular velocity
        ], dim=0)  # Total: 3 + 4 + 3 + 3 = 13
        
        # Ensure float32 dtype
        root_state = root_state.to(dtype=torch.float32)
        
        return root_state
    
    def _extract_targets(self, flat_output):
        """
        Extract target values from flat_output for LeePositionController.
        
        Parameters:
            flat_output: AQB flat_output dict with trajectory information
            
        Returns:
            tuple: (target_pos, target_vel, target_acc, target_yaw) as torch.Tensors
        """
        # Extract targets from flat_output
        target_pos = torch.tensor(flat_output['x'], dtype=torch.float32, device=self.device)
        target_vel = torch.tensor(flat_output['x_dot'], dtype=torch.float32, device=self.device)
        target_acc = torch.tensor(flat_output['x_ddot'], dtype=torch.float32, device=self.device)
        target_yaw = torch.tensor(flat_output['yaw'], dtype=torch.float32, device=self.device)
        
        # Ensure target_yaw has shape (1,) - controller expects last dimension to be 1
        if target_yaw.dim() == 0:
            target_yaw = target_yaw.unsqueeze(0)  # (1,)
        elif target_yaw.shape[-1] != 1:
            target_yaw = target_yaw.unsqueeze(-1)  # Ensure last dim is 1
        
        return target_pos, target_vel, target_acc, target_yaw
    
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
        thrusts = (normalized_cmds + 1) / 2 * self.max_thrusts
        motor_speeds = thrusts / self.k_eta
        motor_speeds = np.sign(motor_speeds) * np.sqrt(np.abs(motor_speeds))

        motor_speeds = np.concatenate([
                motor_speeds[1:2], 
                motor_speeds[0:1], 
                motor_speeds[3:4], 
                motor_speeds[2:3]
                ], axis=0)
        
        return motor_speeds
    
    def update(self, t, state, flat_output, trajectory):
        """
        Main update function called by AdaptiveQuadBench simulator.
        
        Parameters:
            t: current time in seconds
            state: dict with keys x, v, q, w
            flat_output: dict with trajectory information
            
        Returns:
            control_input: dict with cmd_motor_speeds key
        """
        # Convert AQB state to root_state format
        root_state = self._aqb_state_to_root_state(state)
        
        # Extract targets from flat_output
        target_pos, target_vel, target_acc, target_yaw = self._extract_targets(flat_output)
        
        # Call controller forward method
        # The controller expects root_state to have batch dimension, so add it
        root_state_batch = root_state.unsqueeze(0)  # (1, 13)
        target_pos_batch = target_pos.unsqueeze(0)  # (1, 3)
        target_vel_batch = target_vel.unsqueeze(0)  # (1, 3)
        target_acc_batch = target_acc.unsqueeze(0)  # (1, 3)
        target_yaw_batch = target_yaw.unsqueeze(0)  # (1, 1) or (1,)
        
        with torch.no_grad():
            normalized_cmds = self.controller.forward(
                root_state_batch,
                target_pos=target_pos_batch,
                target_vel=target_vel_batch,
                target_acc=target_acc_batch,
                target_yaw=target_yaw_batch,
                body_rate=True  # ang_vel is already in body frame
            )
        
        # Remove batch dimension
        normalized_cmds = normalized_cmds.squeeze(0)  # (num_rotors,)
        
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

        # import pdb; pdb.set_trace()
        
        return control_input

