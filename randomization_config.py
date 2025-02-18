from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from enum import Enum, auto
import numpy as np
from rotorpy.trajectories.random_motion_prim_traj import RapidTrajectory
from rotorpy.wind.dryden_winds import DrydenGust

class ExperimentType(Enum):
    """Enum for different experiment types"""
    WIND = 'wind'
    UNCERTAINTY = 'uncertainty'
    FORCE = 'force'
    TORQUE = 'torque'
    ALL = 'all'  # Combines all disturbances
    
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
    
    def create_ext_force(self) -> Optional[np.ndarray]:
        """Generate randomized external forces"""
        if not self.ext_force_enabled:
            return None
        return np.random.uniform(*self.force_range, size=(self.num_trials, 3))
    
    def create_ext_torque(self) -> Optional[np.ndarray]:
        """Generate randomized external torques"""
        if not self.ext_torque_enabled:
            return None
        return np.random.uniform(*self.torque_range, size=(self.num_trials, 3))
    
    @classmethod
    def from_experiment_type(cls, experiment_type: Union[str, ExperimentType], num_trials: int, seed: Optional[int] = None) -> 'RandomizationConfig':
        """Factory method to create config based on experiment type"""
        if isinstance(experiment_type, str):
            experiment_type = ExperimentType.from_string(experiment_type)
            
        base_config = {
            'num_trials': num_trials,
            'seed': seed,
            'wind_enabled': False,
            'controller_uncertainty_enabled': False,
            'ext_force_enabled': False,
            'ext_torque_enabled': False
        }
        
        configs = {
            ExperimentType.WIND: {'wind_enabled': True},
            ExperimentType.UNCERTAINTY: {'controller_uncertainty_enabled': True},
            ExperimentType.FORCE: {'ext_force_enabled': True},
            ExperimentType.TORQUE: {'ext_torque_enabled': True},
            ExperimentType.ALL: {
                'wind_enabled': True,
                'controller_uncertainty_enabled': True,
                'ext_force_enabled': True,
                'ext_torque_enabled': True
            }
        }
        
        base_config.update(configs[experiment_type])
        return cls(**base_config) 