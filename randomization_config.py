from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from enum import Enum, auto
import numpy as np
from rotorpy.trajectories.random_motion_prim_traj import RapidTrajectory
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.wind.dryden_winds import DrydenGust

class ExperimentType(Enum):
    """Enum for different experiment types"""
    NO = 'no'  # No randomization or disturbances
    WIND = 'wind'
    UNCERTAINTY = 'uncertainty'
    FORCE = 'force'
    TORQUE = 'torque'
    ROTOR_EFFICIENCY = 'rotoreff'
    LATENCY = 'latency'
    PAYLOAD = 'payload'
    ALL = 'all'  # Combines all disturbances
    
    # TODO latency
    # TODO easy swap trajectory type 
    # TODO when to fail feature
    # TODO single visualizer for single trial
    # TODO debug indi-a
    
    @classmethod
    def from_string(cls, value: str) -> 'ExperimentType':
        """Convert string to ExperimentType, case-insensitive"""
        try:
            return cls(value.lower())
        except ValueError:
            valid_types = [e.value for e in cls]
            raise ValueError(f"Invalid experiment type: {value}. Must be one of {valid_types}")
    
    def __str__(self):
        return self.value

@dataclass
class RandomizationConfig:
    """Configuration class for all randomization parameters"""
    num_trials: int
    quad_params: Dict[str, Any]  # Add quad_params as a field
    seed: Optional[int] = None
    
    # Trajectory parameters
    traj_pos_range: tuple = (-2, 2)
    traj_vel_range: tuple = (-2, 2)
    traj_acc_range: tuple = (-2, 2)
    traj_time: float = 5.0
    
    # Wind parameters
    wind_enabled: bool = True
    wind_speed_range: tuple = (-3, 3)
    wind_sigma_range: tuple = (30, 60)
    
    # External force/torque parameters
    ext_force_enabled: bool = True
    ext_torque_enabled: bool = True
    force_range: tuple = (-1, 1)
    torque_range: tuple = (-1, 1)
    
    # Controller uncertainty parameters
    controller_uncertainty_enabled: bool = False
    mass_uncertainty: float = 0.1  # 10% variation
    inertia_uncertainty: float = 0.1
    
    # Add rotor efficiency parameters
    rotor_efficiency_enabled: bool = False
    rotor_efficiency_range: tuple = (0.7, 1.0)  # 70% to 100% efficiency
    
    # Payload parameters (now relative to vehicle parameters)
    payload_enabled: bool = False
    payload_mass_ratio_range: tuple = (0.1, 0.5)  # 10% to 50% of quad mass
    payload_offset_ratio_range: tuple = (-0.2, 0.2)  # -20% to 20% of arm length
    
    def __post_init__(self):
        """Initialize random seed if provided"""
        if self.seed is not None:
            np.random.seed(self.seed)
    
    def create_trajectories(self) -> List[RapidTrajectory]:
        """Generate randomized trajectories"""
        trajectories = []
        for _ in range(self.num_trials):
            pos0 = np.array([0, 0, 0])
            vel0 = np.array([0, 0, 0])
            acc0 = np.array([0, 0, 0])
            gravity = np.array([0, 0, -9.81])
            
            trajectory = RapidTrajectory(pos0, vel0, acc0, gravity)
            posf = np.random.uniform(*self.traj_pos_range, size=3)
            velf = np.random.uniform(*self.traj_vel_range, size=3)
            accf = np.random.uniform(*self.traj_acc_range, size=3)
            
            trajectory.set_goal_position(posf)
            trajectory.set_goal_velocity(velf)
            trajectory.set_goal_acceleration(accf)
            trajectory.generate(self.traj_time)
            trajectories.append(trajectory)
        return trajectories
    
    def create_wind_profiles(self) -> List[Optional[DrydenGust]]:
        """Generate randomized wind profiles"""
        if not self.wind_enabled:
            return [None] * self.num_trials
            
        wind_profiles = []
        for _ in range(self.num_trials):
            wx = np.random.uniform(*self.wind_speed_range)
            wy = np.random.uniform(*self.wind_speed_range)
            wz = np.random.uniform(*self.wind_speed_range)
            sx = np.random.uniform(*self.wind_sigma_range)
            sy = np.random.uniform(*self.wind_sigma_range)
            sz = np.random.uniform(*self.wind_sigma_range)
            
            wind_profile = DrydenGust(
                dt=1/100,
                avg_wind=np.array([wx, wy, wz]),
                sig_wind=np.array([sx, sy, sz])
            )
            wind_profiles.append(wind_profile)
        return wind_profiles
    
    def create_vehicle_params(self, base_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate vehicle parameters with optional rotor efficiency variations"""
        vehicle_params_list = []
        for _ in range(self.num_trials):
            params = base_params.copy()
            
            # Apply rotor efficiency to vehicle parameters
            if self.rotor_efficiency_enabled:
                params['rotor_efficiency'] = np.random.uniform(*self.rotor_efficiency_range, size=4)
            
            vehicle_params_list.append(params)
        return vehicle_params_list

    def create_controllers(self, controller_type: str, base_params: Dict[str, Any], controller_factory) -> List[Any]:
        """Generate controllers with optional parameter uncertainty"""
        controllers = []
        for _ in range(self.num_trials):
            params = base_params.copy()
            if self.controller_uncertainty_enabled:
                params['Ixx'] *= (1 + np.random.normal(0, self.inertia_uncertainty))
                params['Iyy'] *= (1 + np.random.normal(0, self.inertia_uncertainty))
                params['Izz'] *= (1 + np.random.normal(0, self.inertia_uncertainty))
                if 'mass' in params:
                    params['mass'] *= (1 + np.random.normal(0, self.mass_uncertainty))
            controllers.append(controller_factory(controller_type, params))
        return controllers
    
    def create_payload_disturbance(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate coupled force and torque from payload using stored quad_params"""
        if not self.payload_enabled:
            return None, None
            
        payload_forces = []
        payload_torques = []
        gravity = np.array([0, 0, -9.81])
        
        # Use stored quad_params
        quad_mass = self.quad_params.get('mass', 1.0)
        quad_arm_length = self.quad_params.get('arm_length', 0.3)
        
        for _ in range(self.num_trials):
            mass_ratio = np.random.uniform(*self.payload_mass_ratio_range)
            mass = mass_ratio * quad_mass
            
            offset_ratio = np.random.uniform(*self.payload_offset_ratio_range, size=3)
            offset = offset_ratio * quad_arm_length
            
            force = mass * gravity
            torque = np.cross(offset, force)
            
            payload_forces.append(force)
            payload_torques.append(torque)
            
        return np.array(payload_forces), np.array(payload_torques)
    
    def create_ext_force(self) -> Optional[np.ndarray]:
        """Generate randomized external forces"""
        forces = []
        
        if self.ext_force_enabled:
            forces.append(np.random.uniform(*self.force_range, size=(self.num_trials, 3)))
            
        if self.payload_enabled:
            payload_forces, _ = self.create_payload_disturbance()
            forces.append(payload_forces)
            
        if not forces:
            return None
            
        return sum(forces) if len(forces) > 1 else forces[0]
    
    def create_ext_torque(self) -> Optional[np.ndarray]:
        """Generate randomized external torques"""
        torques = []
        
        if self.ext_torque_enabled:
            torques.append(np.random.uniform(*self.torque_range, size=(self.num_trials, 3)))
            
        if self.payload_enabled:
            _, payload_torques = self.create_payload_disturbance()
            torques.append(payload_torques)
            
        if not torques:
            return None
            
        return sum(torques) if len(torques) > 1 else torques[0]
    
    @classmethod
    def from_experiment_type(cls, experiment_type: Union[str, ExperimentType], num_trials: int, quad_params: Dict[str, Any], seed: Optional[int] = None) -> 'RandomizationConfig':
        """Factory method to create config based on experiment type"""
        if isinstance(experiment_type, str):
            experiment_type = ExperimentType.from_string(experiment_type)
            
        base_config = {
            'num_trials': num_trials,
            'quad_params': quad_params,  # Add quad_params
            'seed': seed,
            'wind_enabled': False,
            'controller_uncertainty_enabled': False,
            'ext_force_enabled': False,
            'ext_torque_enabled': False,
            'rotor_efficiency_enabled': False,
            'payload_enabled': False
        }
        
        configs = {
            ExperimentType.NO: {},  # No additional randomization
            ExperimentType.WIND: {'wind_enabled': True},
            ExperimentType.UNCERTAINTY: {'controller_uncertainty_enabled': True},
            ExperimentType.FORCE: {'ext_force_enabled': True},
            ExperimentType.TORQUE: {'ext_torque_enabled': True},
            ExperimentType.ROTOR_EFFICIENCY: {'rotor_efficiency_enabled': True},
            ExperimentType.PAYLOAD: {'payload_enabled': True},  # Add payload case
            ExperimentType.ALL: {
                'wind_enabled': True,
                'controller_uncertainty_enabled': True,
                'ext_force_enabled': True,
                'ext_torque_enabled': True,
                'rotor_efficiency_enabled': True,
                'payload_enabled': True  # Include in ALL case
            }
        }
        
        base_config.update(configs[experiment_type])
        return cls(**base_config) 