from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from enum import Enum, auto
import numpy as np
from rotorpy.trajectories.random_motion_prim_traj import RapidTrajectory
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import CircularTraj
from rotorpy.wind.dryden_winds import DrydenGust

    
# TODO latency
# TODO model uncertainty type (Uniform and Scale)
# TODO tune base controller
# TODO more elegant on rotor efficiency rather than just hard scale cmd_spd
# TODO parallel of NN controller (to pytorch)
class ExperimentType(Enum):
    """Enum for different experiment types"""
    NO = 'no'  # No randomization or disturbances
    WIND = 'wind'
    UNCERTAINTY = 'uncertainty'
    FORCE = 'force'
    TORQUE = 'torque'
    ROTOR_EFFICIENCY = 'rotoreff'
    PAYLOAD = 'payload'

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

class TrajectoryType(Enum):
    """Enum for different trajectory types"""
    RANDOM = 'random'  # Current random motion primitive
    HOVER = 'hover'    # Hover at a point
    CIRCLE = 'circle'  # Circular trajectory
    
    @classmethod
    def from_string(cls, value: str) -> 'TrajectoryType':
        """Convert string to TrajectoryType, case-insensitive"""
        try:
            return cls(value.lower())
        except ValueError:
            valid_types = [e.value for e in cls]
            raise ValueError(f"Invalid trajectory type: {value}. Must be one of {valid_types}")
    
    def __str__(self):
        return self.value

@dataclass
class RandomizationConfig:
    """Configuration class for all randomization parameters"""
    num_trials: int
    quad_params: Dict[str, Any]
    seed: Optional[int] = None
    experiment_type: ExperimentType = ExperimentType.NO
    trajectory_type: TrajectoryType = TrajectoryType.RANDOM
    
    # Base ranges (without intensity scaling)
    base_wind_speed_range: tuple = (-3, 3)
    base_force_range: tuple = (-1, 1)
    base_torque_range: tuple = (-1, 1)
    base_mass_uncertainty: float = 0.1  # 10% variation
    base_inertia_uncertainty: float = 0.1
    base_rotor_efficiency_range: tuple = (0.7, 1.0)
    base_payload_mass_ratio_range: tuple = (0.1, 0.5)
    
    # Current ranges (with intensity scaling)
    wind_speed_range: tuple = (-3, 3)
    force_range: tuple = (-1, 1)
    torque_range: tuple = (-1, 1)
    mass_uncertainty: float = 0.1
    inertia_uncertainty: float = 0.1
    rotor_efficiency_range: tuple = (0.7, 1.0)
    payload_mass_ratio_range: tuple = (0.1, 0.5)
    
    # Fixed ranges (not affected by intensity)
    wind_sigma_range: tuple = (30, 60)
    payload_offset_ratio_range: tuple = (-0.2, 0.2)
    traj_pos_range: tuple = (-2, 2)
    traj_vel_range: tuple = (-2, 2)
    traj_acc_range: tuple = (-2, 2)
    traj_time: float = 5.0

    # Enable flags
    wind_enabled: bool = False
    ext_force_enabled: bool = False
    ext_torque_enabled: bool = False
    controller_uncertainty_enabled: bool = False
    rotor_efficiency_enabled: bool = False
    payload_enabled: bool = False

    @classmethod
    def from_experiment_type(cls, experiment_type: Union[str, ExperimentType], 
                           num_trials: int, 
                           quad_params: Dict[str, Any], 
                           seed: Optional[int] = None,
                           trajectory_type: Union[str, TrajectoryType] = TrajectoryType.RANDOM) -> 'RandomizationConfig':
        """Factory method to create config"""
        if isinstance(experiment_type, str):
            experiment_type = ExperimentType.from_string(experiment_type)
        if isinstance(trajectory_type, str):
            trajectory_type = TrajectoryType.from_string(trajectory_type)
            
        # Create base configuration
        base_config = {
            'num_trials': num_trials,
            'quad_params': quad_params,
            'seed': seed,
            'wind_enabled': False,
            'controller_uncertainty_enabled': False,
            'ext_force_enabled': False,
            'ext_torque_enabled': False,
            'rotor_efficiency_enabled': False,
            'payload_enabled': False,
            'experiment_type': experiment_type,
            'trajectory_type': trajectory_type
        }
        
        # Update enabled flags based on experiment type
        configs = {
            ExperimentType.NO: {},
            ExperimentType.WIND: {'wind_enabled': True},
            ExperimentType.UNCERTAINTY: {'controller_uncertainty_enabled': True},
            ExperimentType.FORCE: {'ext_force_enabled': True},
            ExperimentType.TORQUE: {'ext_torque_enabled': True},
            ExperimentType.ROTOR_EFFICIENCY: {'rotor_efficiency_enabled': True},
            ExperimentType.PAYLOAD: {'payload_enabled': True},
         
        }
        
        base_config.update(configs[experiment_type])
        
        # Create config instance
        config = cls(**base_config)
        
        return config
    

    def scale_ranges_with_intensity(self, intensity: float):
        """Scale the ranges based on intensity and experiment type"""
        if intensity <= 0:
            raise ValueError("Intensity must be positive")
        
        if intensity == 1.0:
            return
            
        if self.experiment_type == ExperimentType.WIND:
            # Only scale wind speed
            max_wind = self.base_wind_speed_range[1] * intensity
            self.wind_speed_range = (-max_wind, max_wind)
            
        elif self.experiment_type == ExperimentType.FORCE:
            # Only scale force
            max_force = self.base_force_range[1] * intensity
            self.force_range = (-max_force, max_force)
            
        elif self.experiment_type == ExperimentType.TORQUE:
            # Only scale torque
            max_torque = self.base_torque_range[1] * intensity
            self.torque_range = (-max_torque, max_torque)
            
        elif self.experiment_type == ExperimentType.UNCERTAINTY:
            # Only scale uncertainties
            self.mass_uncertainty = self.base_mass_uncertainty * intensity
            self.inertia_uncertainty = self.base_inertia_uncertainty * intensity
            
        elif self.experiment_type == ExperimentType.ROTOR_EFFICIENCY:
            # Only scale rotor efficiency
            min_efficiency = max(0.1, self.base_rotor_efficiency_range[0] / intensity)
            self.rotor_efficiency_range = (min_efficiency, 1.0)
            
        elif self.experiment_type == ExperimentType.PAYLOAD:
            # Only scale payload mass
            max_payload = min(2.0, self.base_payload_mass_ratio_range[1] * intensity)
            self.payload_mass_ratio_range = (self.base_payload_mass_ratio_range[0], max_payload)
            
    def create_base_components(self) -> Dict[str, Any]:
        """Generate components that should stay constant across intensity variations"""
        base_components = {
            'trajectories': self.create_trajectories(),
        }

        # Add components based on experiment type
        if self.experiment_type in [ExperimentType.NO, ExperimentType.WIND, ExperimentType.FORCE, ExperimentType.TORQUE, ExperimentType.PAYLOAD]:
            # For these types, vehicle and controller params stay constant
            base_components['vehicle_params'] = self.create_vehicle_params(self.quad_params)
            base_components['controller_params'] = self.create_controller_params(self.quad_params)
        
        return base_components

    def create_varied_components(self) -> Dict[str, Any]:
        """Generate components that vary with intensity based on experiment type"""
        varied_components = {}

        varied_components['wind_profiles'] = self.create_wind_profiles()
        varied_components['ext_force'] = self.create_ext_force()
        varied_components['ext_torque'] = self.create_ext_torque()
        if self.experiment_type == ExperimentType.ROTOR_EFFICIENCY or self.experiment_type == ExperimentType.UNCERTAINTY:
            varied_components['vehicle_params'] = self.create_vehicle_params(self.quad_params)
            varied_components['controller_params'] = self.create_controller_params(self.quad_params)
            
        return varied_components


    def create_trajectories(self) -> List[Union[RapidTrajectory, HoverTraj, CircularTraj]]:
        """Generate trajectories based on specified type"""
        trajectories = []
        for _ in range(self.num_trials):
            if self.trajectory_type == TrajectoryType.RANDOM:
                # Current random motion primitive
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
                
            elif self.trajectory_type == TrajectoryType.HOVER:
                # Hover at random point
                trajectory = HoverTraj()
                
            elif self.trajectory_type == TrajectoryType.CIRCLE:
                
                trajectory = CircularTraj(radius=2)
            else:
                raise ValueError(f"Invalid trajectory type: {self.trajectory_type}")
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
        if not self.ext_force_enabled:
            return None
        return np.random.uniform(*self.force_range, size=(self.num_trials, 3))
    
    def create_ext_torque(self) -> Optional[np.ndarray]:
        """Generate randomized external torques"""
        if not self.ext_torque_enabled:
            return None
        return np.random.uniform(*self.torque_range, size=(self.num_trials, 3))
    
    def create_controller_params(self, base_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate controller parameters with uncertainty"""
        controller_params_list = []
        for _ in range(self.num_trials):
            params = base_params.copy()
            if self.controller_uncertainty_enabled:
                params['Ixx'] *= (1 + np.random.normal(0, self.inertia_uncertainty))
                params['Iyy'] *= (1 + np.random.normal(0, self.inertia_uncertainty))
                params['Izz'] *= (1 + np.random.normal(0, self.inertia_uncertainty))
                if 'mass' in params:
                    params['mass'] *= (1 + np.random.normal(0, self.mass_uncertainty))
            controller_params_list.append(params)
        return controller_params_list

    def __post_init__(self):
        """Initialize random seed if provided"""
        if self.seed is not None:
            np.random.seed(self.seed)
    