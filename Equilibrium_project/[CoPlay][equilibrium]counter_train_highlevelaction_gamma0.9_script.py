import os
import sys
import time
import random
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure

# =========================
# 你需要修改的路径
# =========================

POLICY_POOL_PATH = r"../co_play_partner_pool_counter"

import gym_macro_overcooked
from gym_macro_overcooked.items import (
    Tomato, Lettuce, Onion, Plate, Knife, Delivery, Agent, Food, DirtyPlate
)

# =========================
# 锁线程数，防止并行库抢核
# =========================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# =========================
# 全局随机种子
# =========================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ITEMNAME = [
    "space", "counter", "agent", "tomato", "lettuce", "plate",
    "knife", "delivery", "onion", "dirtyplate", "badlettuce"
]

macroActionDict = {
    "stay": 0,
    "get lettuce 1": 1,
    "get lettuce 2": 2,
    "get plate 1": 3,
    "get plate 2": 4,
    "go to knife 1": 5,
    "deliver 1": 6,
    "chop": 7,
    "go to counter": 8,
    "right": 9,
    "down": 10,
    "left": 11,
    "up": 12
}


def check_action_benevolence(env, action_up, action_down,
                             firsttime_down_go_to_counter,
                             firsttime_up_get_counter_lettuce):

    agent_up = env.agent[0]
    agent_down = env.agent[1]

    counter1_x = 2
    counter1_y = 2

    counter2_x = 2
    counter2_y = 3

    counter3_x = 2
    counter3_y = 4

    counter4_x = 2
    counter4_y = 5

    counter1 = ITEMNAME[env.map[counter1_x][counter1_y]]
    counter2 = ITEMNAME[env.map[counter2_x][counter2_y]]
    counter3 = ITEMNAME[env.map[counter3_x][counter3_y]]
    counter4 = ITEMNAME[env.map[counter4_x][counter4_y]]

    reward_shaping_bonus = 0
    total_reward_bonus = 0

    reward_bonus_up = 0
    reward_bonus_down = 0

    counters = [counter1, counter2, counter3, counter4]

    if any(counter in ("lettuce") for counter in counters):
        best_action = intelligently_find_item_number(env, agent_up, "get lettuce")

        if firsttime_up_get_counter_lettuce is True:
            reward_shaping_bonus = check_benevolence(env, best_action, action_up)
            if reward_shaping_bonus == 20:
                total_reward_bonus += reward_shaping_bonus
                reward_bonus_up = 1000
                firsttime_up_get_counter_lettuce = False

    if all(counter not in ("lettuce") for counter in counters):
        if agent_down.holding and isinstance(agent_down.holding, Lettuce):
            best_action = "go to counter"

            if firsttime_down_go_to_counter is True:
                reward_shaping_bonus = check_benevolence(env, best_action, action_down)

                if reward_shaping_bonus == 20:
                    total_reward_bonus += reward_shaping_bonus
                    reward_bonus_down = 1000
                    firsttime_down_go_to_counter = False

    return (
        reward_bonus_up,
        reward_bonus_down,
        firsttime_down_go_to_counter,
        firsttime_up_get_counter_lettuce
    )


def find_best_reachable_index(can_reach_1, can_reach_2, can_reach_3,
                              distance_1, distance_2, distance_3):
    reachable_indices = []
    distances = []

    if can_reach_1 != 4:
        reachable_indices.append(0)
        distances.append(distance_1)
    if can_reach_2 != 4:
        reachable_indices.append(1)
        distances.append(distance_2)
    if can_reach_3 != 4:
        reachable_indices.append(2)
        distances.append(distance_3)

    if not reachable_indices:
        return False

    if len(reachable_indices) == 1:
        return reachable_indices[0]

    min_distance_index = reachable_indices[distances.index(min(distances))]
    return min_distance_index


def intelligently_find_item_number(env, agent_item, raw_name):

    if raw_name == "get plate":
        target_x_1, target_y_1 = env._findPOitem(agent_item, macroActionDict[raw_name + " 1"])
        can_reach_1 = env._navigate(agent_item, target_x_1, target_y_1)
        distance_1 = env._calDistance(target_x_1, target_y_1, agent_item.x, agent_item.y)

        target_x_2, target_y_2 = env._findPOitem(agent_item, macroActionDict[raw_name + " 2"])
        can_reach_2 = env._navigate(agent_item, target_x_2, target_y_2)
        distance_2 = env._calDistance(target_x_2, target_y_2, agent_item.x, agent_item.y)

        target_x_3, target_y_3 = env._findPOitem(agent_item, macroActionDict["get dirty plate"])
        can_reach_3 = env._navigate(agent_item, target_x_3, target_y_3)
        distance_3 = env._calDistance(target_x_3, target_y_3, agent_item.x, agent_item.y)

        best_action = "stay"

        min_distance_index = find_best_reachable_index(
            can_reach_1, can_reach_2, can_reach_3,
            distance_1, distance_2, distance_3
        )

        if min_distance_index == 0:
            best_action = raw_name + " 1"
        if min_distance_index == 1:
            best_action = raw_name + " 2"
        if min_distance_index == 2:
            best_action = "get dirty plate"

        return best_action

    target_x_1, target_y_1 = env._findPOitem(agent_item, macroActionDict[raw_name + " 1"])
    can_reach_1 = env._navigate(agent_item, target_x_1, target_y_1)
    distance_1 = env._calDistance(target_x_1, target_y_1, agent_item.x, agent_item.y)

    target_x_2, target_y_2 = env._findPOitem(agent_item, macroActionDict[raw_name + " 2"])
    can_reach_2 = env._navigate(agent_item, target_x_2, target_y_2)
    distance_2 = env._calDistance(target_x_2, target_y_2, agent_item.x, agent_item.y)

    best_action = "stay"
    if can_reach_1 == 4 and can_reach_2 != 4:
        best_action = raw_name + " 2"

    if can_reach_1 != 4 and can_reach_2 == 4:
        best_action = raw_name + " 1"

    if can_reach_1 != 4 and can_reach_2 != 4:
        if distance_1 <= distance_2:
            best_action = raw_name + " 1"
        else:
            best_action = raw_name + " 2"

    return best_action


def check_benevolence(env, best_action, action):
    env.reward = 0
    if action == macroActionDict[best_action] and macroActionDict[best_action] != 0:
        env.reward += 20
    return env.reward


def load_policy_pool(pool_dir):
    """
    从 policy pool 目录下加载所有 agent1 policy。
    每个子文件夹下的 model_1500000.zip 视为一个 policy。
    """
    if not os.path.isdir(pool_dir):
        raise FileNotFoundError(f"POLICY_POOL_PATH 不存在: {pool_dir}")

    models = []
    loaded_paths = []

    for folder_name in sorted(os.listdir(pool_dir)):
        folder_path = os.path.join(pool_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        model_path = os.path.join(folder_path, "model_1500000.zip")
        if os.path.exists(model_path):
            try:
                model = PPO.load(model_path, device="cpu")
                models.append(model)
                loaded_paths.append(model_path)
                print(f"[Load policy] {model_path}")
            except Exception as e:
                print(f"[Skip broken model] {model_path} | Error: {e}")

    if len(models) == 0:
        raise RuntimeError(
            f"在 {pool_dir} 下没有成功加载任何 model_1500000.zip"
        )

    print(f"[Policy pool] Loaded {len(models)} models.")
    return models, loaded_paths


class SingleAgentWrapper(gym.Wrapper):
    """
    A wrapper to extract a single agent's perspective from a multi-agent environment.
    这里只训练一个 agent（通常是 agent0），另一个 agent 用已训练好的 policy。
    """
    def __init__(self, env, agent_index, step_penalty_agent0, helping,
             other_agent_model=None, policy_pool=None):
        super(SingleAgentWrapper, self).__init__(env)
        self.agent_index = agent_index
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.other_agent_model = other_agent_model

        self.step_penalty_agent0 = step_penalty_agent0

        self.firsttime_down_go_to_counter = True
        self.firsttime_up_get_counter_lettuce = True

        self.helping = helping
        self.obs = None

        self.policy_pool = policy_pool

        # 仅用于 debug / callback 可视化
        self.current_partner_idx = None
        self.current_partner_path = None

    def set_partner_model(self, model, model_idx=None, model_path=None):
        self.other_agent_model = model
        self.current_partner_idx = model_idx
        self.current_partner_path = model_path

    def reset(self):
        self.obs = self.env.reset()

        self.firsttime_down_go_to_counter = True
        self.firsttime_up_get_counter_lettuce = True

        # ====== 🔥 每个 episode 切换 partner ======
        if self.policy_pool is not None:
            self.other_agent_model = random.choice(self.policy_pool)

        return self.obs[self.agent_index]

    def step(self, action):
        if hasattr(self.env, "agent"):
            agents = self.env.agent
        elif hasattr(self.env, "agents"):
            agents = self.env.agents
        else:
            agents = self.env.unwrapped.agent

        agent0_previous_location = [agents[0].x, agents[0].y]
        agent1_previous_location = [agents[1].x, agents[1].y]

        actions = [0, 0]

        if self.other_agent_model is None:
            other_agent_action = (np.array([0]), None)
        else:
            # 对方固定 policy，不训练，只 predict
            other_agent_action = self.other_agent_model.predict(
                self.obs[1 - self.agent_index],
                deterministic=False
            )

        actions[self.agent_index] = int(action)
        actions[1 - self.agent_index] = int(other_agent_action[0])

        primary_actions, real_execute_macro_actions = self.env._computeLowLevelActions(actions)

        (
            benevolence_reward_up,
            benevolence_reward_down,
            self.firsttime_down_go_to_counter,
            self.firsttime_up_get_counter_lettuce
        ) = check_action_benevolence(
            self.env,
            real_execute_macro_actions[0],
            real_execute_macro_actions[1],
            self.firsttime_down_go_to_counter,
            self.firsttime_up_get_counter_lettuce
        )

        self.obs, rewards, dones, info = self.env.step(primary_actions)
        self.obs = self.env._get_macro_obs()

        agent0_current_location = [agents[0].x, agents[0].y]
        agent1_current_location = [agents[1].x, agents[1].y]

        step_penalty = self.step_penalty_agent0

        if self.helping is True:
            if self.agent_index == 0:
                if agent0_previous_location != agent0_current_location:
                    total_reward = float(rewards[0] + rewards[1]) - step_penalty + benevolence_reward_up
                else:
                    total_reward = float(rewards[0] + rewards[1]) + benevolence_reward_up
                return self.obs[self.agent_index], total_reward, dones, info

            if self.agent_index == 1:
                if agent1_previous_location != agent1_current_location:
                    total_reward = float(rewards[0] + rewards[1]) - step_penalty + benevolence_reward_down
                else:
                    total_reward = float(rewards[0] + rewards[1]) + benevolence_reward_down
                return self.obs[self.agent_index], total_reward, dones, info
        else:
            if self.agent_index == 0:
                if agent0_previous_location != agent0_current_location:
                    total_reward = float(rewards[0] + rewards[1]) - step_penalty
                else:
                    total_reward = float(rewards[0] + rewards[1])
                return self.obs[self.agent_index], total_reward, dones, info

            if self.agent_index == 1:
                if agent1_previous_location != agent1_current_location:
                    total_reward = float(rewards[0] + rewards[1]) - step_penalty
                else:
                    total_reward = float(rewards[0] + rewards[1])
                return self.obs[self.agent_index], total_reward, dones, info


class EpisodeRewardCallback(BaseCallback):
    def __init__(self, save_path, save_freq=100000, verbose=0):
        super(EpisodeRewardCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.step_counter = 0
        self.episode_rewards = []
        self.current_episode_reward = 0.0

        self.start_time = time.time()  # 🔥 新增：记录开始时间

        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        self.step_counter += 1
        self.current_episode_reward += float(self.locals["rewards"][0])

        # episode 结束
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

        # ====== 每 save_freq step 执行 ======
        if self.step_counter % self.save_freq == 0:

            # ====== 保存模型 ======
            model_path = os.path.join(self.save_path, f"model_{self.step_counter}.zip")
            self.model.save(model_path)

            # ====== 时间统计（🔥新增）======
            elapsed = time.time() - self.start_time
            speed = self.step_counter / max(elapsed, 1e-8)

            print(f"[🕒 Time] Step {self.step_counter} | 累计用时: {format_time(elapsed)}")
            print(f"[⚡ Speed] {speed:.1f} steps/sec")
            print(f"Step {self.step_counter}: Model saved at {model_path}")

            # ====== 原有绘图逻辑（完全保留）======
            if len(self.episode_rewards) >= 1:
                window = min(100, len(self.episode_rewards))
                moving_avg = [
                    sum(self.episode_rewards[max(0, i - window + 1): i + 1]) /
                    (i - max(0, i - window + 1) + 1)
                    for i in range(len(self.episode_rewards))
                ]

                plt.figure(figsize=(10, 5))
                plt.plot(moving_avg, label=f"Moving Avg (last {window} episodes)")
                plt.xlabel("Episode")
                plt.ylabel("Average Reward")
                plt.title("Training Progress")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_path, f"avg_reward_{self.step_counter}.png"))
                plt.close()

        return True



def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}分{secs}秒"


def train_one_combo(step_penalty_agent0: int, 
                    helping0: bool, 
                    policy_pool, policy_paths):
    rewardList = [{
        "minitask finished": 0,
        "minitask failed": 0,
        "metatask finished": 0,
        "metatask failed": 0,
        "goodtask finished": 10,
        "goodtask failed": 0,
        "subtask finished": 20,
        "subtask failed": 0,
        "correct delivery": 200,
        "wrong delivery": -50,
        "step penalty": -1,
        "penalize using dirty plate": 0,
        "penalize using bad lettuce": 0,
        "pick up bad lettuce": 0
    }, {
        "minitask finished": 0,
        "minitask failed": 0,
        "metatask finished": 0,
        "metatask failed": 0,
        "goodtask finished": 10,
        "goodtask failed": 0,
        "subtask finished": 20,
        "subtask failed": 0,
        "correct delivery": 200,
        "wrong delivery": -50,
        "step penalty": -1,
        "penalize using dirty plate": 0,
        "penalize using bad lettuce": 0,
        "pick up bad lettuce": 0
    }]

    mac_env_id = "Overcooked-MA-equilibrium-v1"
    env_params = {
        "grid_dim": [5, 8],
        "task": ["lettuce salad"],
        "rewardList": rewardList,
        "map_type": "counter",
        "n_agent": 2,
        "obs_radius": 0,
        "mode": "vector",
        "debug": True
    }

    combo_tag = (
        f"a0sp_{step_penalty_agent0}_"
        f"helping0_{helping0}_"
        f"gamma0.9"
    )

    log_dir = os.path.join("logs", combo_tag)
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["csv", "tensorboard"])

    save_dir_agent0 = os.path.join(
        "final_trained_models",
        f"[coplay][equilibrium][counter]agent0_{combo_tag}"
    )
    os.makedirs(save_dir_agent0, exist_ok=True)

    reward_callback_0 = EpisodeRewardCallback(save_dir_agent0, save_freq=100000)

    shared_env = gym.make(mac_env_id, **env_params)
    shared_env.seed(SEED)
    if hasattr(shared_env, "action_space"):
        shared_env.action_space.seed(SEED)
    if hasattr(shared_env, "observation_space"):
        try:
            shared_env.observation_space.seed(SEED)
        except Exception:
            pass

    # 这里只训练 agent0，agent1 由 policy pool 提供
    env_agent_0 = SingleAgentWrapper(
        shared_env,
        agent_index=0,
        step_penalty_agent0=step_penalty_agent0,
        helping=helping0,
        other_agent_model=None,
        policy_pool=policy_pool   # 🔥 关键
    )

    ppo_params0 = {
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.9,
        "gae_lambda": 0.95,
        "clip_range": 0.3,
        "ent_coef": 0.02,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 0,
    }

    policy_kwargs = dict(net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])])

    model_agent_0 = PPO(
        "MlpPolicy",
        env_agent_0,
        policy_kwargs=policy_kwargs,
        seed=SEED,
        device="cpu",
        **ppo_params0,
    )

    model_agent_0.set_logger(new_logger)

    total_train_steps = 1_500_000

    global_start_time = time.time()
    print(f"\n==================== Training combo: {combo_tag} ====================")




    model_agent_0.learn(
        total_timesteps=total_train_steps,
        callback=reward_callback_0
    )

    phase_end_time = time.time()
    total_duration = phase_end_time - global_start_time
    print(f"[🕒 累计训练时间] {format_time(total_duration)}")



def main():
    # 先统一加载一次 agent1 policy pool，后续所有组合复用
    policy_pool, policy_paths = load_policy_pool(POLICY_POOL_PATH)

    helping0 = [True, False]
    # helping1 = [True, False]
    step_penalty_list_agent0 = [0, 1, 3]
    # step_penalty_list_agent1 = [0, 1, 3]

    for help_partner0 in helping0:
        for sp0 in step_penalty_list_agent0:
            train_one_combo(
                step_penalty_agent0=sp0,
                helping0=help_partner0,
                policy_pool=policy_pool,
                policy_paths=policy_paths
            )


if __name__ == "__main__":
    main()