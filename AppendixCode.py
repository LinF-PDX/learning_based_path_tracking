import numpy as np
import gymnasium as gym
from gymnasium import spaces
import highway_env
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import os
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.graphics import VehicleGraphics
import time
from scipy.linalg import solve_discrete_are

EGO_CAR_SPEED = 7
TRAINING_TIMESTEPS = 500
MAP_SELECTION = "racetrack-v0"
MODEL_SAVE_PATH = "TrainedRLPathTrackingModel"
USE_ROTATING_MAPS = False
TRAINING_MAPS = ["highway-v0", "racetrack-v0"]

def calculate_frenet_frame_errors(env):
    base = env.unwrapped
    ego = base.vehicle
    road = base.road.network

    lane = road.get_lane(road.get_closest_lane_index(ego.position))

    longitudinal_position, lateral_error = lane.local_coordinates(ego.position)
    path_heading = lane.heading_at(longitudinal_position)
    heading_error = ((ego.heading - path_heading + np.pi) % (2*np.pi)) - np.pi
    return lateral_error, heading_error

def calculate_frenet_frame_errors_ghost(base_env, position, heading, return_kappa=False):
    road = base_env.road.network
    lane = road.get_lane(road.get_closest_lane_index(position))

    longitudinal_position, lateral_error = lane.local_coordinates(position)
    path_heading = lane.heading_at(longitudinal_position)

    heading_error = ((heading - path_heading + np.pi) % (2*np.pi)) - np.pi

    if not return_kappa:
        return lateral_error, heading_error
    
    ds = 1.0 
    s2 = (longitudinal_position + ds) % lane.length
    psi2 = lane.heading_at(s2)
    dpsi = (psi2 - path_heading + np.pi) % (2*np.pi) - np.pi
    kappa = dpsi / ds

    return lateral_error, heading_error, kappa

def init_ghost(env, v_ghost=EGO_CAR_SPEED, start_advance=15.0, dt_default=1/15.0):
    base = env.unwrapped
    ego = base.vehicle
    road = base.road
    net = road.network

    lane = net.get_lane(net.get_closest_lane_index(ego.position))

    long_ego, _ = lane.local_coordinates(ego.position)
    x_ghost, y_ghost = lane.position((long_ego + start_advance) % lane.length, 0)
    psi_ghost = lane.heading_at((long_ego + start_advance) % lane.length)

    ghost = Vehicle(road, position=np.array([x_ghost, y_ghost]), heading=psi_ghost, speed=v_ghost)

    ghost.LENGTH = 5
    ghost.color = (225, 0, 0)

    ghost.check_collisions = False
    ghost.collidable = False
    ghost.COLLISIONS_ENABLED = False
    ghost.crashed = False
    ghost.speed_index = 0
    ghost.handle_collisions = lambda *args, **kwargs: None

    road.vehicles.append(ghost)

    controller = LQRController(v_ref=v_ghost, L=ghost.LENGTH, dt=dt_default, delta_max=np.deg2rad(50))
    controller.reset()

    return ghost, controller, ghost.LENGTH

def update_ghost(env, ghost, controller, L, dt=1/15.0):
    base = env.unwrapped
    road = base.road

    lat_err_g, heading_err_g, kappa_ref = calculate_frenet_frame_errors_ghost(base_env = base, position = ghost.position, heading = ghost.heading, return_kappa=True)

    obs_g = np.array([lat_err_g / 2.0, heading_err_g / np.pi, 0.0], dtype=np.float32)

    delta_g = float(controller.compute_action(obs_g, kappa_ref=kappa_ref, v_current=ghost.speed)[0])

    ghost.heading += (ghost.speed / L) * np.tan(delta_g) * dt
    ghost.position[0] += ghost.speed * np.cos(ghost.heading) * dt
    ghost.position[1] += ghost.speed * np.sin(ghost.heading) * dt

    ghost.lane_index = road.network.get_closest_lane_index(ghost.position)

class LossCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.history = {"actor": [], "critic": [], "step": []}

    def _on_step(self):
        if "train/actor_loss" in self.logger.name_to_value:
            self.history["actor"].append(self.logger.name_to_value["train/actor_loss"])
            self.history["critic"].append(self.logger.name_to_value["train/critic_loss"])
            self.history["step"].append(self.num_timesteps)
        return True

    def plot(self):
        if not self.history["step"]: return
        plt.figure()
        plt.plot(self.history["step"], self.history["critic"], label="Critic Loss", alpha=0.3)
        plt.xlabel("Timesteps")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("ddpg_training_loss.png")

class RewardCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        self.timestep_rewards = []

    def _on_step(self):
        if "rewards" in self.locals:
            reward = float(self.locals["rewards"][0])
            self.current_episode_reward += reward
            self.timestep_rewards.append(reward)
            
        if "dones" in self.locals and self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
        return True

    def plot_timesteps(self):
        if not self.timestep_rewards: return
        plt.figure()
        plt.plot(self.timestep_rewards, label="Reward per Timestep", alpha=0.4)
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.title("Training Reward per Timestep")
        plt.legend()
        plt.savefig("ddpg_training_timestep_rewards.png")

class PIDController:
    def __init__(self, Ky, Kpsi, Ki, delta_max, dt, i_limit, last_delta, Kd):
        self.Ky = Ky
        self.Kpsi = Kpsi
        self.Ki = Ki
        self.delta_max = delta_max
        self.dt = dt
        self.integral_ey = 0.0
        self.i_limit = i_limit
        self.last_delta = last_delta
        self.Kd = Kd

    def reset(self):
        self.integral_ey = 0.0
        self.last_delta = 0.0

    def compute_steering(self, lateral_error, heading_error):
        self.integral_ey += lateral_error * self.dt
        self.integral_ey = float(np.clip(self.integral_ey, -self.i_limit, self.i_limit))

        delta = -(self.Ky * lateral_error + self.Kpsi * heading_error + self.Ki * self.integral_ey)

        delta_change = delta - self.last_delta
        delta -= self.Kd * delta_change

        self.last_delta = delta

        delta = float(np.clip(delta, -self.delta_max, self.delta_max))

        return delta

class LQRController:
    def __init__(self, v_ref=EGO_CAR_SPEED, L=5.0, dt=1/15.0, delta_max=np.deg2rad(50), lateral_error_normalization=2.0, heading_error_normalization=np.pi):
        self.v_ref = v_ref
        self.L = L
        self.dt = dt
        self.delta_max = delta_max

        self.lateral_error_normalization = lateral_error_normalization
        self.heading_error_normalization = heading_error_normalization

        self.K = LQR_gain(v_ref, L, dt)

    def reset(self):
        pass

    def compute_action(self, obs, kappa_ref=None, v_current=None):
        norm_lateral_error, norm_heading_error, _ = obs

        e_y = float(norm_lateral_error * self.lateral_error_normalization)
        e_psi = float(norm_heading_error * self.heading_error_normalization)

        x = np.array([[e_y],
                      [e_psi]], dtype=float)

        v_for_lqr = self.v_ref if v_current is None else float(v_current)
        K = LQR_gain(v_for_lqr, self.L, self.dt)

        delta_fb = float((-K @ x).item())

        if kappa_ref is not None:
            delta_ff = 2*(self.L*kappa_ref) 
        else:
            delta_ff = 0.0
        
        delta = delta_ff + delta_fb
        delta = np.clip(delta, -self.delta_max, self.delta_max)

        return np.array([delta], dtype=np.float32)
    

def LQR_gain(v_ref, L, dt):
    A = np.array([[0.0, v_ref], [0.0, 0.0]], dtype=float)

    B = np.array([[0.0], [v_ref / L]], dtype=float)

    A_d = np.eye(2) + A * dt
    B_d = B * dt

    Q = np.diag([5.0, 2.0]) 
    R = np.array([[0.01]])

    P = solve_discrete_are(A_d, B_d, Q, R)

    K = np.linalg.inv(B_d.T @ P @ B_d + R) @ (B_d.T @ P @ A_d)
    
    return K  # shape (1,2)


class CalibratedSteeringConstSpeed(gym.ActionWrapper):
    def __init__(self, env, v_ref=4.0):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.last_steer = 0.0
        self._idx_steer, self._idx_accel = 1, 0
        self.v_ref = v_ref

    def action(self, a):
        self.last_steer = np.clip(float(a[0]), -1.0, 1.0)

        v_current = self.env.unwrapped.vehicle.speed
        kp = 0.5
        acc_cmd = np.clip(kp * (self.v_ref - v_current), -0.5, 0.5)

        act = np.zeros(2, dtype=np.float32)
        act[self._idx_steer] = self.last_steer
        act[self._idx_accel] = acc_cmd
        return act


class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        observation_lower_bounds = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        observation_upper_bounds = np.array([ 1.0,  1.0,  1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=observation_lower_bounds, high=observation_upper_bounds, dtype=np.float32)

    def observation(self, _obs):
        lateral_error, heading_error = calculate_frenet_frame_errors(self.env)
        previous_steering_command = getattr(self.env, "last_steer", 0.0)
        
        lateral_error_normalization = 2  
        heading_error_normalization = np.pi
        
        normalized_lateral_error = lateral_error / lateral_error_normalization
        normalized_heading_error = heading_error / heading_error_normalization
        normalized_previous_steering = float(previous_steering_command)
        
        obs = np.array([normalized_lateral_error, normalized_heading_error, normalized_previous_steering], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, weight_lateral_error=1.0, weight_heading_error=5.0, weight_steering_magnitude=0, weight_steering_change=2):
        super().__init__(env)
        self.weight_lateral_error = weight_lateral_error
        self.weight_heading_error = weight_heading_error
        self.weight_steering_magnitude = weight_steering_magnitude
        self.weight_steering_change = weight_steering_change

    def step(self, action):
        previous_steering_command = getattr(self.env, "last_steer", 0.0)
        current_steering_command = np.clip(float(np.asarray(action).ravel()[0]), -1.0, 1.0)
        
        observation, _, terminated, truncated, info = self.env.step(action)
        lateral_error, heading_error = calculate_frenet_frame_errors(self.env)
        
        tracking_error_penalty = self.weight_lateral_error * abs(lateral_error) + self.weight_heading_error * abs(heading_error)
        steering_magnitude_penalty = self.weight_steering_magnitude * abs(current_steering_command)
        steering_change_penalty = self.weight_steering_change * (abs(current_steering_command - previous_steering_command))
        
        is_off_road = not self.env.unwrapped.vehicle.on_road
        
        if is_off_road:
            off_road_penalty = 10.0 
        else:
            off_road_penalty = 0
        
        reward = - (tracking_error_penalty + steering_magnitude_penalty + steering_change_penalty + off_road_penalty)
        
        return observation, float(reward), terminated, truncated, info


class RotatingMapWrapper(gym.Wrapper):
    def __init__(self, map_ids, v_ref=8.0, rotate_every_n_steps=1000):
        self.map_ids = map_ids
        self.v_ref = v_ref
        self.current_index = 0
        self.step_count = 0
        self.rotate_every_n_steps = rotate_every_n_steps
        self.cfg = {
            "action": {"type": "ContinuousAction"},
            "lanes_count": 2,
            "duration": 30,
            "vehicles_density": 1e-6,
            "offscreen_rendering": True,
            "initial_lane_id": 0,
            "offroad_terminal": True, 
        }
        env = self._make_wrapped_env(map_ids[0])
        super().__init__(env)
    
    def _make_wrapped_env(self, map_id):
        env = gym.make(map_id, render_mode=None, config=self.cfg)
        env = CalibratedSteeringConstSpeed(env, v_ref=self.v_ref)
        env = RewardWrapper(env)
        env = ObsWrapper(env)
        return env
    
    def step(self, action):
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.step_count >= self.rotate_every_n_steps:
            truncated = True
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        if self.step_count >= self.rotate_every_n_steps:
            self.step_count = 0
            self.current_index = (self.current_index + 1) % len(self.map_ids)
            print(f"Rotating map to: {self.map_ids[self.current_index]}")
            self.env.close()
            self.env = self._make_wrapped_env(self.map_ids[self.current_index])
        return self.env.reset(**kwargs)

class PIDSteeringController:
    def __init__(self, Ky = 1, Kpsi =30, Ki = 1, dt = 1/15.0, delta_max = np.deg2rad(50), i_limit = 0.3, last_delta = 0.0, Kd = 0.1):
        self.controller = PIDController(Ky, Kpsi, Ki, delta_max, dt, i_limit, last_delta, Kd)
        self.lateral_error_normalization = 2.0
        self.heading_error_normalization = np.pi

    def reset(self):
        self.controller.reset()

    def compute_action(self, obs):
        norm_lateral_error, norm_heading_error, _ = obs
        lateral_error = float(norm_lateral_error * self.lateral_error_normalization)
        heading_error = float(norm_heading_error * self.heading_error_normalization)
        delta = self.controller.compute_steering(lateral_error, heading_error)
        return np.array([delta], dtype=np.float32)


def create_training_environment(map_id: str = "racetrack-v0", v_ref=8.0, use_rotating_maps=False, training_maps=None):
    if use_rotating_maps and training_maps:
        return RotatingMapWrapper(map_ids=training_maps, v_ref=v_ref)
    
    cfg = {
        "action": {"type": "ContinuousAction"},
        "lanes_count": 1,
        "duration": 30,
        "vehicles_density": 1e-6,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "initial_vehicle_speed": v_ref,
        "other_vehicles_velocity": 20.0,
        "offscreen_rendering": True,
        "initial_lane_id": 0,
        "offroad_terminal": True  
    }
    
    env = gym.make(map_id, render_mode=None, config=cfg)
    env = CalibratedSteeringConstSpeed(env, v_ref=v_ref)
    env = RewardWrapper(env)
    env = ObsWrapper(env)
    
    return env


def create_evaluation_environment(map_id: str = "racetrack-v0", v_ref=8.0):
    cfg = {
        "action": {"type": "ContinuousAction"},
        "lanes_count": 1,
        "duration": 1_000_000_000,
        "offroad_terminal": False, 
        "vehicles_density": 1e-6,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "initial_vehicle_speed": v_ref,
        "other_vehicles_velocity": 20.0,
        "screen_width": 900, "screen_height": 300,
        "offscreen_rendering": False,
        "initial_lane_id": 0,
    }
    env = gym.make(map_id, render_mode="human", config=cfg)
    env = CalibratedSteeringConstSpeed(env, v_ref=v_ref)
    env = RewardWrapper(env)
    env = ObsWrapper(env)
    return env

def main(total_timesteps=TRAINING_TIMESTEPS, save_path=MODEL_SAVE_PATH,
         v_ref=EGO_CAR_SPEED, load_path=None):

    train_env = create_training_environment(
        map_id=MAP_SELECTION, 
        v_ref=EGO_CAR_SPEED, 
        use_rotating_maps=USE_ROTATING_MAPS,
        training_maps=TRAINING_MAPS
    )
    noise = NormalActionNoise(mean=np.array([0.0], dtype=np.float32), sigma=np.array([0.05], dtype=np.float32))
    
    if load_path is None and os.path.exists(f"{save_path}.zip"):
        load_path = save_path
    
    if load_path:
        print(f"Loading existing model: {load_path}.zip")
        model = DDPG.load(load_path, env=train_env)
        model.action_noise = noise
    else:
        model = DDPG("MlpPolicy", train_env, action_noise=noise, verbose=0, policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[256, 256]))) 
    
    loss_cb = LossCallback()
    reward_cb = RewardCallback()
    
    print(f"Starting training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, callback=[ProgressBarCallback(), loss_cb, reward_cb]) 
    loss_cb.plot()
    reward_cb.plot_timesteps()

    model.save(save_path) 
    
    env = create_evaluation_environment(map_id=MAP_SELECTION, v_ref=EGO_CAR_SPEED)
    model = DDPG.load(save_path, env=env)
    
    obs, _ = env.reset()

    ghost, ghost_controller, ghost_L = init_ghost(env, v_ghost=EGO_CAR_SPEED, start_advance=15.0, dt_default=1/15.0)

    start_time = time.time()
    max_duration = 30.0
    time_history = []
    lateral_error_history = []
    heading_error_history = []
    ghost_lateral_error_history = []
    ghost_heading_error_history = []
    try:
        while True:
            env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            update_ghost(env, ghost, ghost_controller, ghost_L)
            
            lateral_error_g, heading_error_g = calculate_frenet_frame_errors_ghost(base_env = env.unwrapped, position = ghost.position, heading = ghost.heading)
            ghost_lateral_error_history.append(lateral_error_g)
            ghost_heading_error_history.append(heading_error_g)
            
            lateral_error, heading_error = calculate_frenet_frame_errors(env)
            time_history.append(time.time() - start_time)
            lateral_error_history.append(lateral_error)
            heading_error_history.append(heading_error)
                        
            print("Time: {:.2f}s, LatErr: {:.4f}m, HeadErr: {:.4f}rad".format(time.time() - start_time, lateral_error, heading_error))
            
            if time.time() - start_time > max_duration:
                break
            
    except KeyboardInterrupt:
        pass
    env.close()
    time_history = np.array(time_history)
    lateral_error_history = np.array(lateral_error_history)
    heading_error_history = np.array(heading_error_history)
    lateral_error_g_history = np.array(ghost_lateral_error_history)
    heading_error_g_history = np.array(ghost_heading_error_history)
    
    rms_lateral_error = np.sqrt(np.mean(lateral_error_history**2))
    rms_heading_error = np.sqrt(np.mean(heading_error_history**2))
    print("RMS Lateral Error (DDPG): {:.4f} m".format(rms_lateral_error))
    print("RMS Heading Error (DDPG): {:.4f} rad".format(rms_heading_error))
    rms_lateral_error_g = np.sqrt(np.mean(lateral_error_g_history**2))
    rms_heading_error_g = np.sqrt(np.mean(heading_error_g_history**2))
    print("RMS Lateral Error (LQR): {:.4f} m".format(rms_lateral_error_g))
    print("RMS Heading Error (LQR): {:.4f} rad".format(rms_heading_error_g))
    
    np.savetxt("time_history.txt", time_history)
    np.savetxt("ddpg_lateral_error_history.txt", lateral_error_history)
    np.savetxt("ddpg_heading_error_history.txt", heading_error_history)
    np.savetxt("lqr_lateral_error_history.txt", lateral_error_g_history)
    np.savetxt("lqr_heading_error_history.txt", heading_error_g_history)

    plt.figure()
    plt.plot(time_history, lateral_error_history, color = 'red',label='DDPG Lateral Error')
    plt.plot(time_history, lateral_error_g_history, color = 'blue',label='LQR Lateral Error')
    plt.xlabel("Time [s]")
    plt.ylabel("Lateral error [m]")
    plt.title("Lateral error vs time (DDPG vs LQR)")
    plt.grid(True)
    plt.legend()
    plt.savefig('Lateral error vs time.png', dpi=300, bbox_inches='tight')

    plt.figure()
    plt.plot(time_history, heading_error_history, color = 'red',label='DDPG Heading Error')
    plt.plot(time_history, heading_error_g_history, color = 'blue',label='LQR Heading Error')
    plt.xlabel("Time [s]")
    plt.ylabel("Heading error [rad]")
    plt.title("Heading error vs time (DDPG vs LQR)")
    plt.grid(True)
    plt.legend()
    plt.savefig('Heading error vs time.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()