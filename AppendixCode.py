"""
LQR FIXES SUMMARY (applied 2026-04-14):
========================================
1. Feedforward gain factor of 2 removed:
   Was:  delta_ff = 2 * L * kappa_ref
   Now:  delta_ff = L * kappa_ref
   Reason: The correct steady-state bicycle-model feedforward is delta = L * kappa.
           The factor of 2 was an incorrect manual fudge that over-steered into curves.

2. Q/R matrices updated via quick auto-tuning (tune_lqr_qr()):
   Was:  Q = diag(5, 2), R = 0.01
   Candidates: diag(5,2)/0.01  |  diag(10,5)/0.10  |  diag(20,10)/0.05
   Best is selected automatically at runtime and printed to console.
   Larger error weights penalise tracking errors more heavily; increased R damps
   oscillatory control activity.

3. Ghost warmup period (WARMUP_STEPS = 30 steps ~ 2 s):
   Errors not recorded until ghost has settled, avoiding inflated RMS from the
   initialisation transient.

4. Curvature estimation step: ds 1.0 -> 0.5 m (LQR_DS constant).
   Coarser finite differences over-estimated kappa on tight racetrack curves,
   causing excessive feedforward steering.
"""

import csv
import os
import time as wallclock

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from scipy.linalg import solve_discrete_are
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from stable_baselines3.common.noise import NormalActionNoise

import highway_env  # noqa: F401 – registers envs as side-effect
from highway_env.vehicle.kinematics import Vehicle

# ── top-level constants ───────────────────────────────────────────────────────
EGO_CAR_SPEED        = 7
TRAINING_TIMESTEPS   = 150_000
MAP_SELECTION        = "racetrack-v0"
MODEL_SAVE_PATH      = "TrainedRLPathTrackingModel"
USE_ROTATING_MAPS    = False
TRAINING_MAPS        = ["highway-v0", "racetrack-v0"]
LOAD_EXISTING_MODELS = False
ALGOS_TO_RUN         = ["ddpg", "ppo"]
SEEDS                = list(range(5))           # 5 repeated trials (seeds 0-4)

# Headless evaluation parameters
EVAL_DT      = 1 / 15.0                    # seconds per simulation step
EVAL_SECONDS = 30.0                         # evaluation window [s]
EVAL_STEPS   = int(EVAL_SECONDS / EVAL_DT)  # 450 steps
WARMUP_STEPS = int(2.0 / EVAL_DT)          # 30 steps ~ 2 s ghost warmup

# LQR tuning defaults (best Q/R selected at runtime by tune_lqr_qr())
LQR_Q  = np.diag([10.0, 5.0])
LQR_R  = np.array([[0.1]])
LQR_DS = 0.5   # curvature finite-difference step [m]

# Q/R candidates evaluated during auto-tuning
QR_CANDIDATES = [
    (np.diag([5.0,  2.0]),  np.array([[0.01]])),   # original
    (np.diag([10.0, 5.0]),  np.array([[0.10]])),   # candidate 1
    (np.diag([20.0, 10.0]), np.array([[0.05]])),   # candidate 2
]


# ── Frenet-frame helpers ──────────────────────────────────────────────────────

def calculate_frenet_frame_errors(env):
    base = env.unwrapped
    ego  = base.vehicle
    road = base.road.network

    lane = road.get_lane(road.get_closest_lane_index(ego.position, ego.heading))

    longitudinal_position, lateral_error = lane.local_coordinates(ego.position)
    path_heading  = lane.heading_at(longitudinal_position)
    heading_error = ((ego.heading - path_heading + np.pi) % (2 * np.pi)) - np.pi
    return lateral_error, heading_error


def calculate_frenet_frame_errors_ghost(base_env, position, heading, return_kappa=False):
    road = base_env.road.network
    lane = road.get_lane(road.get_closest_lane_index(position, heading))

    longitudinal_position, lateral_error = lane.local_coordinates(position)
    path_heading  = lane.heading_at(longitudinal_position)
    heading_error = ((heading - path_heading + np.pi) % (2 * np.pi)) - np.pi

    if not return_kappa:
        return lateral_error, heading_error

    # Fix 4: use LQR_DS (0.5 m) instead of 1.0 m for better kappa on tight curves
    ds  = LQR_DS
    s2  = (longitudinal_position + ds) % lane.length
    psi2 = lane.heading_at(s2)
    dpsi = (psi2 - path_heading + np.pi) % (2 * np.pi) - np.pi
    kappa = dpsi / ds

    return lateral_error, heading_error, kappa


# ── Ghost vehicle helpers ─────────────────────────────────────────────────────

def init_ghost(env, v_ghost=EGO_CAR_SPEED, start_advance=15.0, dt_default=1 / 15.0):
    base = env.unwrapped
    ego  = base.vehicle
    road = base.road
    net  = road.network

    lane     = net.get_lane(net.get_closest_lane_index(ego.position, ego.heading))
    long_ego, _ = lane.local_coordinates(ego.position)

    x_ghost, y_ghost = lane.position((long_ego + start_advance) % lane.length, 0)
    psi_ghost        = lane.heading_at((long_ego + start_advance) % lane.length)

    ghost = Vehicle(road, position=np.array([x_ghost, y_ghost]),
                    heading=psi_ghost, speed=v_ghost)
    ghost.LENGTH = 5
    ghost.color  = (225, 0, 0)
    ghost.check_collisions       = False
    ghost.collidable             = False
    ghost.COLLISIONS_ENABLED     = False
    ghost.crashed                = False
    ghost.speed_index            = 0
    ghost.handle_collisions      = lambda *args, **kwargs: None

    road.vehicles.append(ghost)

    controller = LQRController(v_ref=v_ghost, L=ghost.LENGTH,
                               dt=dt_default, delta_max=np.deg2rad(50))
    controller.reset()
    return ghost, controller, ghost.LENGTH


def update_ghost(env, ghost, controller, L, dt=1 / 15.0):
    base = env.unwrapped
    road = base.road

    lat_err_g, heading_err_g, kappa_ref = calculate_frenet_frame_errors_ghost(
        base_env=base, position=ghost.position,
        heading=ghost.heading, return_kappa=True
    )

    obs_g   = np.array([lat_err_g / 2.0, heading_err_g / np.pi, 0.0], dtype=np.float32)
    delta_g = float(controller.compute_action(
        obs_g, kappa_ref=kappa_ref, v_current=ghost.speed)[0])

    ghost.heading    += (ghost.speed / L) * np.tan(delta_g) * dt
    ghost.position[0] += ghost.speed * np.cos(ghost.heading) * dt
    ghost.position[1] += ghost.speed * np.sin(ghost.heading) * dt
    ghost.lane_index   = road.network.get_closest_lane_index(ghost.position, ghost.heading)


# ── Training callbacks ────────────────────────────────────────────────────────

class AlgoLossCallback(BaseCallback):
    """Logs DDPG or PPO training loss metrics. Raw history accessible via .history."""

    def __init__(self, algo_name: str, out_png: str = None):
        super().__init__(verbose=0)
        self.algo_name = algo_name.lower()
        self.out_png   = out_png
        self.history   = {"step": []}

    def _on_step(self):
        d = self.logger.name_to_value
        if self.algo_name == "ddpg":
            keys = ["train/actor_loss", "train/critic_loss"]
        else:
            keys = ["train/policy_gradient_loss", "train/value_loss",
                    "train/entropy_loss", "train/approx_kl"]

        if any(k in d for k in keys):
            self.history["step"].append(self.num_timesteps)
            for k in keys:
                self.history.setdefault(k, []).append(
                    float(d[k]) if k in d else np.nan)
        return True

    def plot(self, out_png: str = None):
        target = out_png or self.out_png
        if not target or not self.history.get("step"):
            return
        steps = self.history["step"]
        fig, ax = plt.subplots()
        for k, vals in self.history.items():
            if k == "step":
                continue
            ax.plot(steps, vals, label=k, alpha=0.6)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Metric")
        ax.set_title(f"{self.algo_name.upper()} training metrics")
        ax.legend()
        ax.grid(True)
        fig.savefig(target, dpi=300, bbox_inches="tight")
        plt.close(fig)


class RewardCallback(BaseCallback):
    """Records per-timestep and per-episode rewards. Raw data accessible via .timestep_rewards."""

    def __init__(self, out_png: str = None):
        super().__init__(verbose=0)
        self.out_png               = out_png
        self.episode_rewards       = []
        self.current_episode_reward = 0.0
        self.timestep_rewards      = []

    def _on_step(self):
        if "rewards" in self.locals:
            reward = float(self.locals["rewards"][0])
            self.current_episode_reward += reward
            self.timestep_rewards.append(reward)
        if "dones" in self.locals and self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
        return True

    def plot_timesteps(self, out_png: str = None):
        target = out_png or self.out_png
        if not target or not self.timestep_rewards:
            return
        fig, ax = plt.subplots()
        ax.plot(self.timestep_rewards, label="Reward per Timestep", alpha=0.5)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Reward")
        ax.set_title("Training Reward per Timestep")
        ax.legend()
        ax.grid(True)
        fig.savefig(target, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ── Controllers ───────────────────────────────────────────────────────────────

class PIDController:
    def __init__(self, Ky, Kpsi, Ki, delta_max, dt, i_limit, last_delta, Kd):
        self.Ky        = Ky
        self.Kpsi      = Kpsi
        self.Ki        = Ki
        self.delta_max = delta_max
        self.dt        = dt
        self.integral_ey = 0.0
        self.i_limit   = i_limit
        self.last_delta = last_delta
        self.Kd        = Kd

    def reset(self):
        self.integral_ey = 0.0
        self.last_delta  = 0.0

    def compute_steering(self, lateral_error, heading_error):
        self.integral_ey += lateral_error * self.dt
        self.integral_ey  = float(np.clip(self.integral_ey, -self.i_limit, self.i_limit))
        delta = -(self.Ky * lateral_error + self.Kpsi * heading_error
                  + self.Ki * self.integral_ey)
        delta -= self.Kd * (delta - self.last_delta)
        self.last_delta = delta
        return float(np.clip(delta, -self.delta_max, self.delta_max))


class LQRController:
    def __init__(
        self,
        v_ref=EGO_CAR_SPEED,
        L=5.0,
        dt=1 / 15.0,
        delta_max=np.deg2rad(50),
        lateral_error_normalization=2.0,
        heading_error_normalization=np.pi,
        Q=None,
        R=None,
    ):
        self.v_ref    = v_ref
        self.L        = L
        self.dt       = dt
        self.delta_max = delta_max
        self.lateral_error_normalization  = lateral_error_normalization
        self.heading_error_normalization  = heading_error_normalization
        # Use provided Q/R or fall back to module-level defaults
        self.Q = Q if Q is not None else LQR_Q.copy()
        self.R = R if R is not None else LQR_R.copy()
        self.K = LQR_gain(v_ref, L, dt, Q=self.Q, R=self.R)

    def reset(self):
        pass

    def compute_action(self, obs, kappa_ref=None, v_current=None):
        norm_lateral_error, norm_heading_error, _ = obs
        e_y   = float(norm_lateral_error * self.lateral_error_normalization)
        e_psi = float(norm_heading_error * self.heading_error_normalization)
        x = np.array([[e_y], [e_psi]], dtype=float)

        v_for_lqr = self.v_ref if v_current is None else float(v_current)
        K = LQR_gain(v_for_lqr, self.L, self.dt, Q=self.Q, R=self.R)
        delta_fb = float((-K @ x).item())

        if kappa_ref is not None:
            # Fix 1: removed incorrect factor of 2.
            # Correct bicycle-model steady-state feedforward: delta_ff = L * kappa_ref
            delta_ff = self.L * kappa_ref
        else:
            delta_ff = 0.0

        delta = np.clip(delta_ff + delta_fb, -self.delta_max, self.delta_max)
        return np.array([delta], dtype=np.float32)


def LQR_gain(v_ref, L, dt, Q=None, R=None):
    if Q is None:
        Q = LQR_Q
    if R is None:
        R = LQR_R
    A   = np.array([[0.0, v_ref], [0.0, 0.0]], dtype=float)
    B   = np.array([[0.0], [v_ref / L]],        dtype=float)
    A_d = np.eye(2) + A * dt
    B_d = B * dt
    P   = solve_discrete_are(A_d, B_d, Q, R)
    K   = np.linalg.inv(B_d.T @ P @ B_d + R) @ (B_d.T @ P @ A_d)
    return K   # shape (1, 2)


# ── Environment wrappers ──────────────────────────────────────────────────────

class CalibratedSteeringConstSpeed(gym.ActionWrapper):
    def __init__(self, env, v_ref=4.0):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.last_steer   = 0.0
        self._idx_steer   = 1
        self._idx_accel   = 0
        self.v_ref        = v_ref

    def action(self, a):
        self.last_steer = np.clip(float(a[0]), -1.0, 1.0)
        v_current = self.env.unwrapped.vehicle.speed
        acc_cmd   = np.clip(0.5 * (self.v_ref - v_current), -0.5, 0.5)
        act = np.zeros(2, dtype=np.float32)
        act[self._idx_steer] = self.last_steer
        act[self._idx_accel] = acc_cmd
        return act


class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low  = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([ 1.0,  1.0,  1.0,  1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, _obs):
        lateral_error, heading_error = calculate_frenet_frame_errors(self.env)
        prev_steer = float(getattr(self.env, "last_steer", 0.0))
        ey_norm = np.clip(lateral_error / 2.0, -1.0, 1.0)
        sin_e   = float(np.sin(heading_error))
        cos_e   = float(np.cos(heading_error))
        return np.clip(
            np.array([ey_norm, sin_e, cos_e, prev_steer], dtype=np.float32),
            -1.0, 1.0,
        )


class RewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        weight_lateral_error=1.0,
        weight_heading_error=5.0,
        weight_steering_magnitude=0.05,
        weight_steering_change=2.0,
        weight_forward_speed=0.4,
        weight_reverse_speed=1.2,
    ):
        super().__init__(env)
        self.weight_lateral_error       = weight_lateral_error
        self.weight_heading_error       = weight_heading_error
        self.weight_steering_magnitude  = weight_steering_magnitude
        self.weight_steering_change     = weight_steering_change
        self.weight_forward_speed       = weight_forward_speed
        self.weight_reverse_speed       = weight_reverse_speed
        self._last_s            = None
        self._last_lane_length  = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        base = self.env.unwrapped
        ego  = base.vehicle
        net  = base.road.network
        lane = net.get_lane(net.get_closest_lane_index(ego.position, ego.heading))
        s, _ = lane.local_coordinates(ego.position)
        self._last_s           = float(s)
        self._last_lane_length = float(lane.length)
        return obs

    def _wrapped_ds(self, s_now, s_prev, lane_length):
        ds = s_now - s_prev
        ds = (ds + 0.5 * lane_length) % lane_length - 0.5 * lane_length
        return float(ds)

    def step(self, action):
        previous_steering  = getattr(self.env, "last_steer", 0.0)
        current_steering   = np.clip(float(np.asarray(action).ravel()[0]), -1.0, 1.0)
        observation, _, terminated, truncated, info = self.env.step(action)

        lateral_error, heading_error = calculate_frenet_frame_errors(self.env)

        base = self.env.unwrapped
        ego  = base.vehicle
        net  = base.road.network
        lane = net.get_lane(net.get_closest_lane_index(ego.position, ego.heading))
        s_now, _ = lane.local_coordinates(ego.position)

        if self._last_s is None:
            self._last_s           = float(s_now)
            self._last_lane_length = float(lane.length)

        lane_length = float(lane.length)
        lane_heading  = lane.heading_at(float(s_now))
        lane_tangent  = np.array([np.cos(lane_heading), np.sin(lane_heading)], dtype=np.float32)
        signed_forward_speed = float(ego.velocity @ lane_tangent)

        self._last_s           = float(s_now)
        self._last_lane_length = lane_length

        tracking_error_penalty   = (self.weight_lateral_error * abs(lateral_error)
                                    + self.weight_heading_error * abs(heading_error))
        steering_magnitude_penalty = self.weight_steering_magnitude * abs(current_steering)
        steering_change_penalty    = self.weight_steering_change * abs(current_steering - previous_steering)
        off_road_penalty           = 10.0 if not self.env.unwrapped.vehicle.on_road else 0.0
        forward_speed_reward       = self.weight_forward_speed * max(signed_forward_speed, 0.0)
        reverse_speed_penalty      = self.weight_reverse_speed * max(-signed_forward_speed, 0.0)

        reward = (forward_speed_reward - reverse_speed_penalty
                  - tracking_error_penalty - steering_magnitude_penalty
                  - steering_change_penalty - off_road_penalty)
        return observation, float(reward), terminated, truncated, info


class RotatingMapWrapper(gym.Wrapper):
    def __init__(self, map_ids, v_ref=8.0, rotate_every_n_steps=1000):
        self.map_ids              = map_ids
        self.v_ref                = v_ref
        self.current_index        = 0
        self.step_count           = 0
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
            self.step_count    = 0
            self.current_index = (self.current_index + 1) % len(self.map_ids)
            print(f"Rotating map to: {self.map_ids[self.current_index]}")
            self.env.close()
            self.env = self._make_wrapped_env(self.map_ids[self.current_index])
        return self.env.reset(**kwargs)


# ── Environment factories ─────────────────────────────────────────────────────

def create_training_environment(map_id="racetrack-v0", v_ref=8.0,
                                use_rotating_maps=False, training_maps=None):
    if use_rotating_maps and training_maps:
        return RotatingMapWrapper(map_ids=training_maps, v_ref=v_ref)
    cfg = {
        "action":                   {"type": "ContinuousAction"},
        "lanes_count":              1,
        "duration":                 30,
        "vehicles_density":         1e-6,
        "other_vehicles_type":      "highway_env.vehicle.behavior.IDMVehicle",
        "initial_vehicle_speed":    v_ref,
        "other_vehicles_velocity":  20.0,
        "offscreen_rendering":      True,
        "initial_lane_id":          0,
        "offroad_terminal":         True,
    }
    env = gym.make(map_id, render_mode=None, config=cfg)
    env = CalibratedSteeringConstSpeed(env, v_ref=v_ref)
    env = RewardWrapper(env)
    env = ObsWrapper(env)
    return env


def create_evaluation_environment(map_id="racetrack-v0", v_ref=8.0):
    """Visual evaluation environment (opens a window). Use for single-run demos."""
    cfg = {
        "action":                  {"type": "ContinuousAction"},
        "lanes_count":             1,
        "duration":                1_000_000_000,
        "offroad_terminal":        False,
        "vehicles_density":        1e-6,
        "other_vehicles_type":     "highway_env.vehicle.behavior.IDMVehicle",
        "initial_vehicle_speed":   v_ref,
        "other_vehicles_velocity": 20.0,
        "screen_width":            900,
        "screen_height":           300,
        "offscreen_rendering":     False,
        "initial_lane_id":         0,
    }
    env = gym.make(map_id, render_mode="human", config=cfg)
    env = CalibratedSteeringConstSpeed(env, v_ref=v_ref)
    env = RewardWrapper(env)
    env = ObsWrapper(env)
    return env


def create_headless_evaluation_environment(map_id="racetrack-v0", v_ref=8.0):
    """Headless evaluation environment for multi-seed runs (no window)."""
    cfg = {
        "action":                  {"type": "ContinuousAction"},
        "lanes_count":             1,
        "duration":                1_000_000_000,
        "offroad_terminal":        False,
        "vehicles_density":        1e-6,
        "other_vehicles_type":     "highway_env.vehicle.behavior.IDMVehicle",
        "initial_vehicle_speed":   v_ref,
        "other_vehicles_velocity": 20.0,
        "offscreen_rendering":     True,
        "initial_lane_id":         0,
    }
    env = gym.make(map_id, render_mode=None, config=cfg)
    env = CalibratedSteeringConstSpeed(env, v_ref=v_ref)
    env = RewardWrapper(env)
    env = ObsWrapper(env)
    return env


# ── Training ──────────────────────────────────────────────────────────────────

def train_algo(algo_name: str, total_timesteps: int, base_save_path: str,
               v_ref: float, seed: int = 0):
    """
    Train one algorithm for one seed.
    Returns (model_save_path, loss_history_dict, timestep_rewards_list).
    Does NOT auto-save per-seed plots; caller assembles aggregate plots later.
    """
    algo      = algo_name.lower()
    save_path = f"{base_save_path}_{algo}_seed{seed}"

    train_env = create_training_environment(
        map_id=MAP_SELECTION, v_ref=v_ref,
        use_rotating_maps=USE_ROTATING_MAPS, training_maps=TRAINING_MAPS,
    )
    train_env.reset(seed=seed)

    loss_cb   = AlgoLossCallback(algo_name=algo)
    reward_cb = RewardCallback()

    if algo == "ddpg":
        noise = NormalActionNoise(
            mean=np.array([0.0], dtype=np.float32),
            sigma=np.array([0.05], dtype=np.float32),
        )
        if LOAD_EXISTING_MODELS and os.path.exists(f"{save_path}.zip"):
            print(f"[DDPG] Loading existing model: {save_path}.zip")
            model = DDPG.load(save_path, env=train_env)
            model.action_noise = noise
        else:
            model = DDPG(
                "MlpPolicy", train_env, action_noise=noise, verbose=0,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], qf=[256, 256])),
                seed=seed,
            )
    elif algo == "ppo":
        if LOAD_EXISTING_MODELS and os.path.exists(f"{save_path}.zip"):
            print(f"[PPO] Loading existing model: {save_path}.zip")
            model = PPO.load(save_path, env=train_env)
        else:
            model = PPO(
                "MlpPolicy", train_env, verbose=0,
                n_steps=256, batch_size=64,
                policy_kwargs=dict(net_arch=[64, 64]),
                seed=seed,
            )
    else:
        raise ValueError(f"Unknown algo: {algo_name}")

    print(f"[{algo.upper()}] Seed {seed} — training {total_timesteps} steps…")
    model.learn(total_timesteps=total_timesteps,
                callback=[ProgressBarCallback(), loss_cb, reward_cb])

    model.save(save_path)
    train_env.close()
    print(f"[{algo.upper()}] Seed {seed} — model saved: {save_path}.zip")
    return save_path, loss_cb.history, reward_cb.timestep_rewards


# ── Evaluation (visual, single-seed) ─────────────────────────────────────────

def evaluate_algo(algo_name: str, model_path: str, v_ref: float,
                  max_duration: float = 30.0):
    """Visual evaluation with an on-screen render window (original behaviour)."""
    algo = algo_name.lower()
    env  = create_evaluation_environment(map_id=MAP_SELECTION, v_ref=v_ref)
    env.reset(seed=0)

    model = DDPG.load(model_path, env=env) if algo == "ddpg" \
        else PPO.load(model_path, env=env)

    obs, _ = env.reset()
    ghost, ghost_controller, ghost_L = init_ghost(
        env, v_ghost=v_ref, start_advance=15.0, dt_default=1 / 15.0)

    start_time = wallclock.time()
    time_history, lateral_error_history, heading_error_history = [], [], []
    ghost_lateral_error_history, ghost_heading_error_history   = [], []

    try:
        while True:
            env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            update_ghost(env, ghost, ghost_controller, ghost_L)

            lat_g, head_g = calculate_frenet_frame_errors_ghost(
                base_env=env.unwrapped, position=ghost.position, heading=ghost.heading)
            ghost_lateral_error_history.append(lat_g)
            ghost_heading_error_history.append(head_g)

            lat, head = calculate_frenet_frame_errors(env)
            t = wallclock.time() - start_time
            time_history.append(t)
            lateral_error_history.append(lat)
            heading_error_history.append(head)
            print(f"[{algo.upper()}] t={t:.2f}s  ey={lat:.4f}m  epsi={head:.4f}rad")

            if t > max_duration:
                break
            if terminated or truncated:
                continue
    except KeyboardInterrupt:
        pass
    env.close()

    time_arr   = np.array(time_history)
    lat_arr    = np.array(lateral_error_history)
    head_arr   = np.array(heading_error_history)
    lat_g_arr  = np.array(ghost_lateral_error_history)
    head_g_arr = np.array(ghost_heading_error_history)

    rms_lat      = float(np.sqrt(np.mean(lat_arr**2)))    if len(lat_arr)    else float("nan")
    rms_head     = float(np.sqrt(np.mean(head_arr**2)))   if len(head_arr)   else float("nan")
    rms_lat_lqr  = float(np.sqrt(np.mean(lat_g_arr**2)))  if len(lat_g_arr)  else float("nan")
    rms_head_lqr = float(np.sqrt(np.mean(head_g_arr**2))) if len(head_g_arr) else float("nan")

    print(f"RMS Lateral Error ({algo.upper()}): {rms_lat:.4f} m")
    print(f"RMS Heading Error ({algo.upper()}): {rms_head:.4f} rad")
    print(f"RMS Lateral Error (LQR): {rms_lat_lqr:.4f} m")
    print(f"RMS Heading Error (LQR): {rms_head_lqr:.4f} rad")

    np.savetxt(f"{algo}_time_history.txt",           time_arr)
    np.savetxt(f"{algo}_lateral_error_history.txt",  lat_arr)
    np.savetxt(f"{algo}_heading_error_history.txt",  head_arr)
    np.savetxt(f"{algo}_lqr_lateral_error_history.txt",  lat_g_arr)
    np.savetxt(f"{algo}_lqr_heading_error_history.txt",  head_g_arr)

    for metric, rl_vals, lqr_vals, ylabel, fname_suffix in [
        ("Lateral", lat_arr, lat_g_arr, "Lateral error [m]",    "lateral_error_vs_time"),
        ("Heading", head_arr, head_g_arr, "Heading error [rad]", "heading_error_vs_time"),
    ]:
        fig, ax = plt.subplots()
        ax.plot(time_arr, rl_vals,  label=f"{algo.upper()} {metric} Error", alpha=0.8)
        ax.plot(time_arr[:len(lqr_vals)], lqr_vals, label=f"LQR {metric} Error", alpha=0.8)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} error vs time ({algo.upper()} vs LQR)")
        ax.grid(True)
        ax.legend()
        fig.savefig(f"{algo}_{fname_suffix}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {
        "algo": algo, "time": time_arr,
        "lat": lat_arr,   "head": head_arr,
        "lat_lqr": lat_g_arr, "head_lqr": head_g_arr,
        "rms_lat": rms_lat,   "rms_head": rms_head,
        "rms_lat_lqr": rms_lat_lqr, "rms_head_lqr": rms_head_lqr,
    }


# ── Evaluation (headless, multi-seed) ────────────────────────────────────────

def evaluate_algo_headless(algo_name: str, model_path: str, v_ref: float,
                            eval_steps: int = EVAL_STEPS,
                            warmup_steps: int = WARMUP_STEPS,
                            seed: int = 0):
    """
    Headless evaluation for the multi-seed loop.
    Uses step-count timing (eval_steps steps after warmup_steps warmup).
    Returns the same dict structure as evaluate_algo().
    """
    algo = algo_name.lower()
    env  = create_headless_evaluation_environment(map_id=MAP_SELECTION, v_ref=v_ref)
    obs, _ = env.reset(seed=seed)

    model = DDPG.load(model_path, env=env) if algo == "ddpg" \
        else PPO.load(model_path, env=env)

    ghost, ghost_controller, ghost_L = init_ghost(
        env, v_ghost=v_ref, start_advance=15.0, dt_default=EVAL_DT)

    time_history, lateral_error_history, heading_error_history = [], [], []
    ghost_lat_history, ghost_head_history = [], []

    total_steps = warmup_steps + eval_steps

    for step in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        update_ghost(env, ghost, ghost_controller, ghost_L, dt=EVAL_DT)

        if step >= warmup_steps:
            t = (step - warmup_steps) * EVAL_DT
            lat,  head  = calculate_frenet_frame_errors(env)
            lat_g, head_g = calculate_frenet_frame_errors_ghost(
                base_env=env.unwrapped, position=ghost.position, heading=ghost.heading)
            time_history.append(t)
            lateral_error_history.append(lat)
            heading_error_history.append(head)
            ghost_lat_history.append(lat_g)
            ghost_head_history.append(head_g)

        if terminated or truncated:
            obs, _ = env.reset()
            ghost, ghost_controller, ghost_L = init_ghost(
                env, v_ghost=v_ref, start_advance=15.0, dt_default=EVAL_DT)

    env.close()

    t_arr      = np.array(time_history)
    lat_arr    = np.array(lateral_error_history)
    head_arr   = np.array(heading_error_history)
    lat_g_arr  = np.array(ghost_lat_history)
    head_g_arr = np.array(ghost_head_history)

    def rms(a): return float(np.sqrt(np.mean(a**2))) if len(a) else float("nan")

    return {
        "algo": algo, "time": t_arr,
        "lat": lat_arr,   "head": head_arr,
        "lat_lqr": lat_g_arr, "head_lqr": head_g_arr,
        "rms_lat": rms(lat_arr), "rms_head": rms(head_arr),
        "rms_lat_lqr": rms(lat_g_arr), "rms_head_lqr": rms(head_g_arr),
    }


# ── LQR Q/R auto-tuning ───────────────────────────────────────────────────────

def tune_lqr_qr(v_ref=EGO_CAR_SPEED, tuning_steps=200, seed=0):
    """
    Evaluate each Q/R candidate with the LQR ghost (no RL agent needed).
    Returns the (Q, R) pair with lowest RMS lateral error and updates the
    module-level LQR_Q and LQR_R globals.
    """
    global LQR_Q, LQR_R

    print("[LQR TUNING] Testing Q/R candidates…")
    best_q, best_r, best_rms = None, None, float("inf")

    for Q_cand, R_cand in QR_CANDIDATES:
        env = create_headless_evaluation_environment(map_id=MAP_SELECTION, v_ref=v_ref)
        obs, _ = env.reset(seed=seed)

        # Create ghost with this specific Q/R
        ghost, _, ghost_L = init_ghost(env, v_ghost=v_ref, start_advance=15.0,
                                       dt_default=EVAL_DT)
        controller = LQRController(v_ref=v_ref, L=ghost_L, dt=EVAL_DT,
                                   Q=Q_cand, R=R_cand)

        lat_errors = []
        for step in range(tuning_steps):
            # Ego uses zero steering (irrelevant — we only care about ghost errors)
            obs, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
            update_ghost(env, ghost, controller, ghost_L, dt=EVAL_DT)
            if terminated or truncated:
                obs, _ = env.reset()
            if step >= WARMUP_STEPS:
                lat_g, _ = calculate_frenet_frame_errors_ghost(
                    base_env=env.unwrapped,
                    position=ghost.position, heading=ghost.heading)
                lat_errors.append(lat_g)

        env.close()
        rms_val = float(np.sqrt(np.mean(np.array(lat_errors) ** 2))) if lat_errors else float("inf")
        label   = f"diag({', '.join(f'{v:.0f}' for v in np.diag(Q_cand))})"
        print(f"  Q={label}, R={R_cand[0, 0]:.4f}  ->  RMS ey={rms_val:.4f} m")

        if rms_val < best_rms:
            best_rms = rms_val
            best_q, best_r = Q_cand.copy(), R_cand.copy()

    LQR_Q = best_q
    LQR_R = best_r
    label = f"diag({', '.join(f'{v:.0f}' for v in np.diag(LQR_Q))})"
    print(f"[LQR TUNING] Best: Q={label}, R={LQR_R[0,0]:.4f}  ->  RMS ey={best_rms:.4f} m\n")
    return best_q, best_r


# ── Aggregate plotting ────────────────────────────────────────────────────────

def _bin_rewards(rewards, bin_size=500):
    """Bin a 1-D reward array into mean values of width bin_size."""
    r = np.asarray(rewards)
    n_bins = len(r) // bin_size
    if n_bins == 0:
        return np.array([0]), np.array([np.mean(r)])
    binned = np.array([np.mean(r[i * bin_size:(i + 1) * bin_size])
                       for i in range(n_bins)])
    x = np.arange(n_bins) * bin_size + bin_size // 2
    return x, binned


def plot_training_reward_curves(all_train_data, algos=None):
    if algos is None:
        algos = ALGOS_TO_RUN
    colors = {"ddpg": "tab:blue", "ppo": "tab:orange"}
    bin_size = 500

    fig, ax = plt.subplots(figsize=(10, 5))
    for algo in algos:
        seed_rewards = [all_train_data[algo][s]["reward"] for s in SEEDS]
        min_len = min(len(r) for r in seed_rewards)
        if min_len == 0:
            continue
        # Bin each seed
        xs_list, binned_list = [], []
        for r in seed_rewards:
            x, b = _bin_rewards(r[:min_len], bin_size=bin_size)
            xs_list.append(x)
            binned_list.append(b)
        n_bins = min(len(b) for b in binned_list)
        x_common = xs_list[0][:n_bins]
        stacked  = np.array([b[:n_bins] for b in binned_list])  # (5, n_bins)
        mean, std = np.mean(stacked, axis=0), np.std(stacked, axis=0)
        c = colors.get(algo, None)
        ax.plot(x_common, mean, label=algo.upper(), color=c)
        ax.fill_between(x_common, mean - std, mean + std, alpha=0.3, color=c)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward (binned mean)")
    ax.set_title("Training Reward (mean ± std over 5 seeds)")
    ax.legend()
    ax.grid(True)
    fig.savefig("training_reward_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved training_reward_curves.png")


def plot_training_loss_curves(all_train_data, algos=None):
    if algos is None:
        algos = ALGOS_TO_RUN

    algo_keys = {
        "ddpg": ["train/actor_loss", "train/critic_loss"],
        "ppo":  ["train/policy_gradient_loss", "train/value_loss"],
    }
    algo_labels = {
        "ddpg": ["Actor Loss", "Critic Loss"],
        "ppo":  ["Policy Gradient Loss", "Value Loss"],
    }
    active = [a for a in algos if a in algo_keys]
    if not active:
        return

    n_rows = len(active)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows), squeeze=False)

    for row, algo in enumerate(active):
        for col, (key, label) in enumerate(zip(algo_keys[algo], algo_labels[algo])):
            ax = axes[row][col]
            all_steps, all_vals = [], []
            for s in SEEDS:
                hist  = all_train_data[algo][s]["loss"]
                steps = np.array(hist.get("step", []))
                vals  = np.array(hist.get(key, []))
                if len(steps) > 1 and len(vals) == len(steps):
                    all_steps.append(steps)
                    all_vals.append(vals)

            if not all_steps:
                ax.set_title(f"{algo.upper()} {label} (no data)")
                continue

            x_min    = max(s[0]  for s in all_steps)
            x_max    = min(s[-1] for s in all_steps)
            x_common = np.linspace(x_min, x_max, 200)
            interped = np.array([np.interp(x_common, s, v)
                                 for s, v in zip(all_steps, all_vals)])
            mean, std = np.mean(interped, axis=0), np.std(interped, axis=0)

            ax.plot(x_common, mean, label=f"Mean", color="tab:blue" if algo == "ddpg" else "tab:orange")
            ax.fill_between(x_common, mean - std, mean + std, alpha=0.3,
                            color="tab:blue" if algo == "ddpg" else "tab:orange")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel(label)
            ax.set_title(f"{algo.upper()} — {label} (mean ± std, 5 seeds)")
            ax.legend()
            ax.grid(True)

    fig.tight_layout()
    fig.savefig("training_loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved training_loss_curves.png")


def _mean_std_timeseries(all_results, algo, key):
    """Stack arrays from all seeds, return (mean, std) along axis=0."""
    arrays = np.array([all_results[algo][s][key] for s in SEEDS])
    return np.mean(arrays, axis=0), np.std(arrays, axis=0)


def plot_combined_eval(all_results, algos=None):
    if algos is None:
        algos = ALGOS_TO_RUN
    t = np.arange(EVAL_STEPS) * EVAL_DT

    colors = {"ddpg": "tab:blue", "ppo": "tab:orange", "lqr": "tab:green"}

    for metric_key, ylabel, fname in [
        ("lat",  "Lateral error [m]",   "combined_lateral_error_vs_time.png"),
        ("head", "Heading error [rad]", "combined_heading_error_vs_time.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for algo in algos:
            mean, std = _mean_std_timeseries(all_results, algo, metric_key)
            c = colors.get(algo)
            ax.plot(t[:len(mean)], mean, label=algo.upper(), color=c)
            ax.fill_between(t[:len(mean)], mean - std, mean + std, alpha=0.3, color=c)

        # LQR from first algo
        ref_algo = algos[0]
        lqr_key  = "lat_lqr" if metric_key == "lat" else "head_lqr"
        mean_lqr, std_lqr = _mean_std_timeseries(all_results, ref_algo, lqr_key)
        ax.plot(t[:len(mean_lqr)], mean_lqr, label="LQR", color=colors["lqr"])
        ax.fill_between(t[:len(mean_lqr)],
                        mean_lqr - std_lqr, mean_lqr + std_lqr,
                        alpha=0.3, color=colors["lqr"])

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split('[')[0].strip()} vs time (mean ± std, 5 seeds)")
        ax.legend()
        ax.grid(True)
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


def plot_per_algo_eval(all_results, algos=None):
    if algos is None:
        algos = ALGOS_TO_RUN
    t = np.arange(EVAL_STEPS) * EVAL_DT

    algo_colors = {"ddpg": "tab:blue", "ppo": "tab:orange"}

    for algo in algos:
        c = algo_colors.get(algo, "tab:blue")
        for metric_key, lqr_key, ylabel, fname in [
            ("lat",  "lat_lqr",  "Lateral error [m]",
             f"{algo}_lateral_error_vs_time.png"),
            ("head", "head_lqr", "Heading error [rad]",
             f"{algo}_heading_error_vs_time.png"),
        ]:
            mean_rl,  std_rl  = _mean_std_timeseries(all_results, algo, metric_key)
            mean_lqr, std_lqr = _mean_std_timeseries(all_results, algo, lqr_key)

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(t[:len(mean_rl)],  mean_rl,  label=algo.upper(), color=c)
            ax.fill_between(t[:len(mean_rl)],
                            mean_rl - std_rl, mean_rl + std_rl, alpha=0.3, color=c)
            ax.plot(t[:len(mean_lqr)], mean_lqr, label="LQR", color="tab:green")
            ax.fill_between(t[:len(mean_lqr)],
                            mean_lqr - std_lqr, mean_lqr + std_lqr,
                            alpha=0.3, color="tab:green")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{algo.upper()} vs LQR — {ylabel.split('[')[0].strip()} "
                         f"(mean ± std, 5 seeds)")
            ax.legend()
            ax.grid(True)
            fig.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {fname}")


# ── CSV summary ───────────────────────────────────────────────────────────────

def save_summary_csv(summary: dict, path="multi_seed_summary.csv"):
    fieldnames = ["controller", "rms_lat_mean", "rms_lat_std",
                  "rms_head_mean", "rms_head_std"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ctrl, stats in summary.items():
            writer.writerow({
                "controller":   ctrl.upper(),
                "rms_lat_mean": f"{stats['mean_lat']:.4f}",
                "rms_lat_std":  f"{stats['std_lat']:.4f}",
                "rms_head_mean": f"{stats['mean_head']:.4f}",
                "rms_head_std":  f"{stats['std_head']:.4f}",
            })
    print(f"Summary saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(total_timesteps=TRAINING_TIMESTEPS, save_path=MODEL_SAVE_PATH,
         v_ref=EGO_CAR_SPEED):

    # Fix 2: select best Q/R for LQR via quick auto-tuning
    tune_lqr_qr(v_ref=v_ref)

    all_results    = {algo: {} for algo in ALGOS_TO_RUN}
    all_train_data = {algo: {} for algo in ALGOS_TO_RUN}

    t_total_start = wallclock.time()

    for algo in ALGOS_TO_RUN:
        for i, seed in enumerate(SEEDS):
            print(f"\n{'='*60}")
            print(f"[{algo.upper()}] Seed {i + 1}/{len(SEEDS)}  (seed={seed})")
            print(f"{'='*60}")

            t_seed_start = wallclock.time()

            model_path, loss_hist, reward_hist = train_algo(
                algo, total_timesteps, save_path, v_ref, seed=seed)

            # Rough ETA after the very first seed
            if algo == ALGOS_TO_RUN[0] and i == 0:
                elapsed_first = wallclock.time() - t_seed_start
                total_runs    = len(ALGOS_TO_RUN) * len(SEEDS)
                print(f"\n  [ETA] First training took {elapsed_first:.0f}s. "
                      f"Estimated total training: "
                      f"{elapsed_first * total_runs / 60:.1f} min "
                      f"({elapsed_first * total_runs / 3600:.2f} h)")

            print(f"[{algo.upper()}] Evaluating seed {seed}…")
            seed_results = evaluate_algo_headless(
                algo, model_path, v_ref, seed=seed)

            all_results[algo][seed]    = seed_results
            all_train_data[algo][seed] = {
                "reward": reward_hist,
                "loss":   loss_hist,
            }

            print(f"[{algo.upper()}] Seed {i + 1}/{len(SEEDS)} complete — "
                  f"RMS ey={seed_results['rms_lat']:.4f}m, "
                  f"RMS epsi={seed_results['rms_head']:.4f}rad  "
                  f"(LQR: ey={seed_results['rms_lat_lqr']:.4f}m, "
                  f"epsi={seed_results['rms_head_lqr']:.4f}rad)")

    # ── Summary statistics ────────────────────────────────────────────────────
    summary = {}
    for algo in ALGOS_TO_RUN:
        lats  = [all_results[algo][s]["rms_lat"]     for s in SEEDS]
        heads = [all_results[algo][s]["rms_head"]    for s in SEEDS]
        summary[algo] = {
            "mean_lat":  np.mean(lats),  "std_lat":  np.std(lats),
            "mean_head": np.mean(heads), "std_head": np.std(heads),
        }

    # LQR standalone stats (from first algo's ghost data)
    ref = ALGOS_TO_RUN[0]
    lats_lqr  = [all_results[ref][s]["rms_lat_lqr"]  for s in SEEDS]
    heads_lqr = [all_results[ref][s]["rms_head_lqr"] for s in SEEDS]
    summary["lqr"] = {
        "mean_lat":  np.mean(lats_lqr),  "std_lat":  np.std(lats_lqr),
        "mean_head": np.mean(heads_lqr), "std_head": np.std(heads_lqr),
    }

    # ── Console summary table ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY  (mean ± std over 5 seeds)")
    print(f"{'='*70}")
    print(f"{'Controller':<10}  {'RMS ey [m]':>22}  {'RMS epsi [rad]':>22}")
    print(f"{'-'*70}")
    for ctrl in list(ALGOS_TO_RUN) + ["lqr"]:
        s = summary[ctrl]
        print(f"{ctrl.upper():<10}  "
              f"{s['mean_lat']:>10.4f} ± {s['std_lat']:<10.4f}  "
              f"{s['mean_head']:>10.4f} ± {s['std_head']:<10.4f}")
    print(f"{'='*70}")

    total_elapsed = wallclock.time() - t_total_start
    print(f"\nTotal wall-clock time: {total_elapsed/60:.1f} min")

    # ── Persist results ───────────────────────────────────────────────────────
    save_summary_csv(summary)

    # ── Aggregate plots ───────────────────────────────────────────────────────
    print("\nGenerating plots…")
    plot_training_reward_curves(all_train_data)
    plot_training_loss_curves(all_train_data)
    plot_combined_eval(all_results)
    plot_per_algo_eval(all_results)

    print("\nAll done.")


if __name__ == "__main__":
    main()
