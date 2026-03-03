import numpy as np
import gymnasium as gym
from gymnasium import spaces
import highway_env
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
import matplotlib.pyplot as plt
import os
from highway_env.vehicle.kinematics import Vehicle
import time
from scipy.linalg import solve_discrete_are

EGO_CAR_SPEED = 7
TRAINING_TIMESTEPS = 4096
MAP_SELECTION = "racetrack-v0"
MODEL_SAVE_PATH = "TrainedRLPathTrackingModel"
USE_ROTATING_MAPS = False
TRAINING_MAPS = ["highway-v0", "racetrack-v0"]

# Run sequentially
ALGOS_TO_RUN = ["ddpg", "ppo"]  # order matters
SEED = 0


def calculate_frenet_frame_errors(env):
    base = env.unwrapped
    ego = base.vehicle
    road = base.road.network

    lane = road.get_lane(road.get_closest_lane_index(ego.position))

    longitudinal_position, lateral_error = lane.local_coordinates(ego.position)
    path_heading = lane.heading_at(longitudinal_position)
    heading_error = ((ego.heading - path_heading + np.pi) % (2 * np.pi)) - np.pi
    return lateral_error, heading_error


def calculate_frenet_frame_errors_ghost(base_env, position, heading, return_kappa=False):
    road = base_env.road.network
    lane = road.get_lane(road.get_closest_lane_index(position))

    longitudinal_position, lateral_error = lane.local_coordinates(position)
    path_heading = lane.heading_at(longitudinal_position)

    heading_error = ((heading - path_heading + np.pi) % (2 * np.pi)) - np.pi

    if not return_kappa:
        return lateral_error, heading_error

    ds = 1.0
    s2 = (longitudinal_position + ds) % lane.length
    psi2 = lane.heading_at(s2)
    dpsi = (psi2 - path_heading + np.pi) % (2 * np.pi) - np.pi
    kappa = dpsi / ds

    return lateral_error, heading_error, kappa


def init_ghost(env, v_ghost=EGO_CAR_SPEED, start_advance=15.0, dt_default=1 / 15.0):
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


def update_ghost(env, ghost, controller, L, dt=1 / 15.0):
    base = env.unwrapped
    road = base.road

    lat_err_g, heading_err_g, kappa_ref = calculate_frenet_frame_errors_ghost(
        base_env=base, position=ghost.position, heading=ghost.heading, return_kappa=True
    )

    obs_g = np.array([lat_err_g / 2.0, heading_err_g / np.pi, 0.0], dtype=np.float32)

    delta_g = float(controller.compute_action(obs_g, kappa_ref=kappa_ref, v_current=ghost.speed)[0])

    ghost.heading += (ghost.speed / L) * np.tan(delta_g) * dt
    ghost.position[0] += ghost.speed * np.cos(ghost.heading) * dt
    ghost.position[1] += ghost.speed * np.sin(ghost.heading) * dt

    ghost.lane_index = road.network.get_closest_lane_index(ghost.position)


class AlgoLossCallback(BaseCallback):
    """
    Logs DDPG or PPO training metrics (so you can keep one callback and not break PPO).
    DDPG: train/actor_loss, train/critic_loss
    PPO : train/policy_gradient_loss, train/value_loss, train/entropy_loss, train/approx_kl
    """

    def __init__(self, algo_name: str, out_png: str):
        super().__init__(verbose=0)
        self.algo_name = algo_name.lower()
        self.out_png = out_png
        self.history = {"step": []}

    def _on_step(self):
        d = self.logger.name_to_value

        if self.algo_name == "ddpg":
            keys = ["train/actor_loss", "train/critic_loss"]
        else:
            keys = ["train/policy_gradient_loss", "train/value_loss", "train/entropy_loss", "train/approx_kl"]

        found_any = any(k in d for k in keys)
        if found_any:
            self.history["step"].append(self.num_timesteps)
            for k in keys:
                if k in d:
                    self.history.setdefault(k, []).append(float(d[k]))
                else:
                    # keep arrays aligned
                    self.history.setdefault(k, []).append(np.nan)
        return True

    def plot(self):
        if len(self.history.get("step", [])) == 0:
            return
        steps = self.history["step"]
        plt.figure()
        for k, vals in self.history.items():
            if k == "step":
                continue
            plt.plot(steps, vals, label=k, alpha=0.6)
        plt.xlabel("Timesteps")
        plt.ylabel("Metric")
        plt.title(f"{self.algo_name.upper()} training metrics")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.out_png, dpi=300, bbox_inches="tight")


class RewardCallback(BaseCallback):
    def __init__(self, out_png: str):
        super().__init__(verbose=0)
        self.out_png = out_png
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
        if not self.timestep_rewards:
            return
        plt.figure()
        plt.plot(self.timestep_rewards, label="Reward per Timestep", alpha=0.5)
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.title("Training Reward per Timestep")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.out_png, dpi=300, bbox_inches="tight")


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
    def __init__(
        self,
        v_ref=EGO_CAR_SPEED,
        L=5.0,
        dt=1 / 15.0,
        delta_max=np.deg2rad(50),
        lateral_error_normalization=2.0,
        heading_error_normalization=np.pi,
    ):
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

        x = np.array([[e_y], [e_psi]], dtype=float)

        v_for_lqr = self.v_ref if v_current is None else float(v_current)
        K = LQR_gain(v_for_lqr, self.L, self.dt)

        delta_fb = float((-K @ x).item())

        if kappa_ref is not None:
            delta_ff = 2 * (self.L * kappa_ref)
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
        # [ey_norm, sin(epsi), cos(epsi), prev_steer]
        low  = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([ 1.0,  1.0,  1.0,  1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, _obs):
        lateral_error, heading_error = calculate_frenet_frame_errors(self.env)
        prev_steer = float(getattr(self.env, "last_steer", 0.0))

        ey_norm = np.clip(lateral_error / 2.0, -1.0, 1.0)
        sin_e = float(np.sin(heading_error))
        cos_e = float(np.cos(heading_error))

        obs = np.array([ey_norm, sin_e, cos_e, prev_steer], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)


class RewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        weight_lateral_error=1.0,
        weight_heading_error=5.0,
        weight_steering_magnitude=0.05,   # <-- small, helps avoid circles
        weight_steering_change=2.0,
        weight_progress=1.0,              # <-- NEW: reward forward progress
        weight_wrong_way=2.0,             # <-- NEW: penalize reverse progress
    ):
        super().__init__(env)
        self.weight_lateral_error = weight_lateral_error
        self.weight_heading_error = weight_heading_error
        self.weight_steering_magnitude = weight_steering_magnitude
        self.weight_steering_change = weight_steering_change
        self.weight_progress = weight_progress
        self.weight_wrong_way = weight_wrong_way

        self._last_s = None
        self._last_lane_length = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        # Initialize last_s on reset so first step doesn't get a crazy delta
        base = self.env.unwrapped
        ego = base.vehicle
        net = base.road.network
        lane = net.get_lane(net.get_closest_lane_index(ego.position))
        s, _ = lane.local_coordinates(ego.position)

        self._last_s = float(s)
        self._last_lane_length = float(lane.length)
        return obs

    def _wrapped_ds(self, s_now: float, s_prev: float, lane_length: float) -> float:
        """
        Wrap delta-s into [-L/2, L/2] so loop tracks behave correctly.
        """
        ds = s_now - s_prev
        # wrap to [-L/2, L/2]
        ds = (ds + 0.5 * lane_length) % lane_length - 0.5 * lane_length
        return float(ds)

    def step(self, action):
        previous_steering_command = getattr(self.env, "last_steer", 0.0)
        current_steering_command = np.clip(float(np.asarray(action).ravel()[0]), -1.0, 1.0)

        observation, _, terminated, truncated, info = self.env.step(action)

        # Tracking errors (same as before)
        lateral_error, heading_error = calculate_frenet_frame_errors(self.env)

        # Progress along lane (s)
        base = self.env.unwrapped
        ego = base.vehicle
        net = base.road.network
        lane = net.get_lane(net.get_closest_lane_index(ego.position))
        s_now, _ = lane.local_coordinates(ego.position)

        # Initialize if somehow missing
        if self._last_s is None:
            self._last_s = float(s_now)
            self._last_lane_length = float(lane.length)

        lane_length = float(lane.length)
        ds = self._wrapped_ds(float(s_now), float(self._last_s), lane_length)

        # Update stored s
        self._last_s = float(s_now)
        self._last_lane_length = lane_length

        # Penalties (same idea as before)
        tracking_error_penalty = (
            self.weight_lateral_error * abs(lateral_error)
            + self.weight_heading_error * abs(heading_error)
        )
        steering_magnitude_penalty = self.weight_steering_magnitude * abs(current_steering_command)
        steering_change_penalty = self.weight_steering_change * abs(current_steering_command - previous_steering_command)

        is_off_road = not self.env.unwrapped.vehicle.on_road
        off_road_penalty = 10.0 if is_off_road else 0.0

        # NEW: progress reward + wrong-way penalty
        # Forward progress: ds > 0
        progress_reward = self.weight_progress * max(ds, 0.0)
        wrong_way_penalty = self.weight_wrong_way * max(-ds, 0.0)

        reward = (
            progress_reward
            - wrong_way_penalty
            - tracking_error_penalty
            - steering_magnitude_penalty
            - steering_change_penalty
            - off_road_penalty
        )

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
        "offroad_terminal": True,
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
        "screen_width": 900,
        "screen_height": 300,
        "offscreen_rendering": False,
        "initial_lane_id": 0,
    }
    env = gym.make(map_id, render_mode="human", config=cfg)
    env = CalibratedSteeringConstSpeed(env, v_ref=v_ref)
    env = RewardWrapper(env)
    env = ObsWrapper(env)
    return env


def train_algo(algo_name: str, total_timesteps: int, base_save_path: str, v_ref: float):
    algo = algo_name.lower()
    save_path = f"{base_save_path}_{algo}"

    train_env = create_training_environment(
        map_id=MAP_SELECTION,
        v_ref=v_ref,
        use_rotating_maps=USE_ROTATING_MAPS,
        training_maps=TRAINING_MAPS,
    )
    train_env.reset(seed=SEED)

    loss_cb = AlgoLossCallback(algo_name=algo, out_png=f"{algo}_training_metrics.png")
    reward_cb = RewardCallback(out_png=f"{algo}_training_timestep_rewards.png")

    if algo == "ddpg":
        noise = NormalActionNoise(mean=np.array([0.0], dtype=np.float32), sigma=np.array([0.05], dtype=np.float32))

        if os.path.exists(f"{save_path}.zip"):
            print(f"[DDPG] Loading existing model: {save_path}.zip")
            model = DDPG.load(save_path, env=train_env)
            model.action_noise = noise
        else:
            model = DDPG(
                "MlpPolicy",
                train_env,
                action_noise=noise,
                verbose=0,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], qf=[256, 256])),
                seed=SEED,
            )

    elif algo == "ppo":
        if os.path.exists(f"{save_path}.zip"):
            print(f"[PPO] Loading existing model: {save_path}.zip")
            model = PPO.load(save_path, env=train_env)
        else:
            model = PPO(
                "MlpPolicy",
                train_env,
                verbose=0,
                n_steps=256,
                batch_size=64,
                policy_kwargs=dict(net_arch=[64, 64]),
                seed=SEED,
            )
    else:
        raise ValueError(f"Unknown algo: {algo_name}")

    print(f"[{algo.upper()}] Starting training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, callback=[ProgressBarCallback(), loss_cb, reward_cb])

    loss_cb.plot()
    reward_cb.plot_timesteps()

    model.save(save_path)
    train_env.close()
    return save_path


def evaluate_algo(algo_name: str, model_path: str, v_ref: float, max_duration: float = 30.0):
    algo = algo_name.lower()

    env = create_evaluation_environment(map_id=MAP_SELECTION, v_ref=v_ref)
    env.reset(seed=SEED)

    if algo == "ddpg":
        model = DDPG.load(model_path, env=env)
    elif algo == "ppo":
        model = PPO.load(model_path, env=env)
    else:
        raise ValueError(f"Unknown algo for eval: {algo_name}")

    obs, _ = env.reset()

    ghost, ghost_controller, ghost_L = init_ghost(env, v_ghost=v_ref, start_advance=15.0, dt_default=1 / 15.0)

    start_time = time.time()
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

            lateral_error_g, heading_error_g = calculate_frenet_frame_errors_ghost(
                base_env=env.unwrapped, position=ghost.position, heading=ghost.heading
            )
            ghost_lateral_error_history.append(lateral_error_g)
            ghost_heading_error_history.append(heading_error_g)

            lateral_error, heading_error = calculate_frenet_frame_errors(env)
            t = time.time() - start_time
            time_history.append(t)
            lateral_error_history.append(lateral_error)
            heading_error_history.append(heading_error)

            print(f"[{algo.upper()}] Time: {t:.2f}s, LatErr: {lateral_error:.4f}m, HeadErr: {heading_error:.4f}rad")

            if t > max_duration:
                break
            if terminated or truncated:
            #obs, _ = env.reset()
            # optionally re-init ghost too:
            # ghost, ghost_controller, ghost_L = init_ghost(env, v_ghost=v_ref, start_advance=15.0, dt_default=1/15.0)
                continue

    except KeyboardInterrupt:
        pass

    env.close()

    time_history = np.array(time_history)
    lateral_error_history = np.array(lateral_error_history)
    heading_error_history = np.array(heading_error_history)
    lateral_error_g_history = np.array(ghost_lateral_error_history)
    heading_error_g_history = np.array(ghost_heading_error_history)

    rms_lateral_error = float(np.sqrt(np.mean(lateral_error_history**2))) if len(lateral_error_history) else float("nan")
    rms_heading_error = float(np.sqrt(np.mean(heading_error_history**2))) if len(heading_error_history) else float("nan")
    rms_lateral_error_g = float(np.sqrt(np.mean(lateral_error_g_history**2))) if len(lateral_error_g_history) else float("nan")
    rms_heading_error_g = float(np.sqrt(np.mean(heading_error_g_history**2))) if len(heading_error_g_history) else float("nan")

    print(f"RMS Lateral Error ({algo.upper()}): {rms_lateral_error:.4f} m")
    print(f"RMS Heading Error ({algo.upper()}): {rms_heading_error:.4f} rad")
    print(f"RMS Lateral Error (LQR): {rms_lateral_error_g:.4f} m")
    print(f"RMS Heading Error (LQR): {rms_heading_error_g:.4f} rad")

    # Save per-algo logs
    np.savetxt(f"{algo}_time_history.txt", time_history)
    np.savetxt(f"{algo}_lateral_error_history.txt", lateral_error_history)
    np.savetxt(f"{algo}_heading_error_history.txt", heading_error_history)
    np.savetxt(f"{algo}_lqr_lateral_error_history.txt", lateral_error_g_history)
    np.savetxt(f"{algo}_lqr_heading_error_history.txt", heading_error_g_history)

    # Per-algo plots vs LQR
    plt.figure()
    plt.plot(time_history, lateral_error_history, label=f"{algo.upper()} Lateral Error", alpha=0.8)
    plt.plot(time_history[: len(lateral_error_g_history)], lateral_error_g_history, label="LQR Lateral Error", alpha=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Lateral error [m]")
    plt.title(f"Lateral error vs time ({algo.upper()} vs LQR)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{algo}_lateral_error_vs_time.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(time_history, heading_error_history, label=f"{algo.upper()} Heading Error", alpha=0.8)
    plt.plot(time_history[: len(heading_error_g_history)], heading_error_g_history, label="LQR Heading Error", alpha=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Heading error [rad]")
    plt.title(f"Heading error vs time ({algo.upper()} vs LQR)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{algo}_heading_error_vs_time.png", dpi=300, bbox_inches="tight")

    return {
        "algo": algo,
        "time": time_history,
        "lat": lateral_error_history,
        "head": heading_error_history,
        "lat_lqr": lateral_error_g_history,
        "head_lqr": heading_error_g_history,
        "rms_lat": rms_lateral_error,
        "rms_head": rms_heading_error,
        "rms_lat_lqr": rms_lateral_error_g,
        "rms_head_lqr": rms_heading_error_g,
    }


def main(total_timesteps=TRAINING_TIMESTEPS, save_path=MODEL_SAVE_PATH, v_ref=EGO_CAR_SPEED):
    results = {}

    # 1) Train + evaluate sequentially
    #for algo in ALGOS_TO_RUN:
    #    model_path = train_algo(algo, total_timesteps=total_timesteps, base_save_path=save_path, v_ref=v_ref)
    #    results[algo] = evaluate_algo(algo, model_path=model_path, v_ref=v_ref, max_duration=30.0)
    
    #results[0] = evaluate_algo('ddpg', model_path="/home/linfu/grad/syde675/project/learning_based_path_tracking/TrainedRLPathTrackingModel_ddpg.zip", v_ref=v_ref, max_duration=30.0)
    #results[1] = evaluate_algo('ppo', model_path="/home/linfu/grad/syde675/project/learning_based_path_tracking/TrainedRLPathTrackingModel_ppo.zip", v_ref=v_ref, max_duration=30.0)

    model_path = train_algo('ppo', total_timesteps=total_timesteps, base_save_path=save_path, v_ref=v_ref)
    results['ppo'] = evaluate_algo('ppo', model_path=model_path, v_ref=v_ref, max_duration=30.0)

    # 2) Combined plots (DDPG vs PPO vs LQR) if both exist
    if "ddpg" in results and "ppo" in results:
        # Use each algo's own time base (they may differ if terminated early)
        plt.figure()
        plt.plot(results["ddpg"]["time"], results["ddpg"]["lat"], label="DDPG Lateral Error", alpha=0.8)
        plt.plot(results["ppo"]["time"], results["ppo"]["lat"], label="PPO Lateral Error", alpha=0.8)
        # Plot one LQR reference (take PPO's, but either is fine)
        plt.plot(results["ppo"]["time"][: len(results["ppo"]["lat_lqr"])], results["ppo"]["lat_lqr"], label="LQR Lateral Error", alpha=0.8)
        plt.xlabel("Time [s]")
        plt.ylabel("Lateral error [m]")
        plt.title("Lateral error vs time (DDPG vs PPO vs LQR)")
        plt.grid(True)
        plt.legend()
        plt.savefig("combined_lateral_error_vs_time.png", dpi=300, bbox_inches="tight")

        plt.figure()
        plt.plot(results["ddpg"]["time"], results["ddpg"]["head"], label="DDPG Heading Error", alpha=0.8)
        plt.plot(results["ppo"]["time"], results["ppo"]["head"], label="PPO Heading Error", alpha=0.8)
        plt.plot(results["ppo"]["time"][: len(results["ppo"]["head_lqr"])], results["ppo"]["head_lqr"], label="LQR Heading Error", alpha=0.8)
        plt.xlabel("Time [s]")
        plt.ylabel("Heading error [rad]")
        plt.title("Heading error vs time (DDPG vs PPO vs LQR)")
        plt.grid(True)
        plt.legend()
        plt.savefig("combined_heading_error_vs_time.png", dpi=300, bbox_inches="tight")

    # 3) Print summary
    print("\n===== Summary =====")
    for algo, r in results.items():
        print(
            f"{algo.upper():>4}: RMS ey={r['rms_lat']:.4f} m, RMS epsi={r['rms_head']:.4f} rad | "
            f"LQR ey={r['rms_lat_lqr']:.4f} m, LQR epsi={r['rms_head_lqr']:.4f} rad"
        )


if __name__ == "__main__":
    main()