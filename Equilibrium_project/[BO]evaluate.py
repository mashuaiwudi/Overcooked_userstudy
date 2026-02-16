import gym
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import matplotlib.pyplot as plt
import time
import random
from gym_macro_overcooked.items import Tomato, Lettuce, Onion, Plate, Knife, Delivery, Agent, Food, DirtyPlate
import time
import cv2
import numpy as np


"""Begin of Some utils"""

class SingleAgentWrapper(gym.Wrapper):
    """
    A wrapper to extract a single agent's perspective from a multi-agent environment.
    """
    def __init__(self, env, agent_index, other_agent_model=None):
        super(SingleAgentWrapper, self).__init__(env)
        self.agent_index = agent_index
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.other_agent_model = other_agent_model
        
        self.obs = None


    def reset(self):
        self.obs = self.env.reset()

        return self.obs[self.agent_index]

    def step(self, action):

        actions = [0, 0]

        other_agent_action = self.other_agent_model.predict(self.obs[1 - self.agent_index])

        actions[self.agent_index] = action

        actions[1 - self.agent_index] = other_agent_action[0]

        primary_actions, _ = self.env._computeLowLevelActions(actions)

        self.obs, rewards, dones, info = self.env.step(primary_actions)

        self.obs = self.env._get_macro_obs()



        return self.obs[self.agent_index], rewards[2], dones, info


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
},{
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



mac_env_id = 'Overcooked-MA-equilibrium-v0'
env_params = {
    'grid_dim': [5, 5],
    'task': ["lettuce salad"],
    'rewardList': rewardList,
    'map_type': "circle",
    'n_agent': 2,
    'obs_radius': 0,
    'mode': "vector",
    'debug': True
}


# Initialize shared environment
shared_env = gym.make(mac_env_id, **env_params)




# Wrap each agent
env_agent_0 = SingleAgentWrapper(shared_env, agent_index=0)
env_agent_1 = SingleAgentWrapper(shared_env, agent_index=1)






# assume parameter = [yes/no, sp_agent0, sp_agent1]


# step penalty → index mapping
STEP_PENALTY_MAP = {
    0: 0,
    -1: 1,
    -10: 10,
    -20: 20,
    -50: 50
}

# yes/no → helping mapping
HELPING_MAP = {
    "yes": True,
    "no": False,
    "Yes": True,
    "No": False,
    True: True,
    False: False
}


def build_policy_path(agent_id, helping, sp_agent0, sp_agent1):
    """
    agent_id: 0 or 1
    helping: yes/no or True/False
    sp_agent0: step penalty of agent0
    sp_agent1: step penalty of agent1
    """

    helping_bool = HELPING_MAP[helping]

    X1 = STEP_PENALTY_MAP[sp_agent0]
    X2 = STEP_PENALTY_MAP[sp_agent1]
    X3 = helping_bool

    path = (
        f"../policy_pool/"
        f"[equilibrium]agent{agent_id}_a0sp_{X1}_a1sp_{X2}_helping_{X3}/"
        f"model_500000"
    )

    return path


"""End of Some utils"""

def evaluate(parameter, evaluate_espisode_length, human_policy):

    helping = parameter[0]
    sp_agent0 = parameter[1]
    sp_agent1 = parameter[2]

    # build paths
    path_agent_ai = build_policy_path(
        agent_id=0,
        helping=helping,
        sp_agent0=sp_agent0,
        sp_agent1=sp_agent1
    )


    print("Loading AI model from:", path_agent_ai)

    # load models
    model_ai = PPO.load(path_agent_ai, env=env_agent_0)

    # reset env
    obs = shared_env.reset()
    obs = shared_env._get_macro_obs()

    reward_this = 0.0

    for step in range(evaluate_espisode_length):

        action_0, _ = model_ai.predict(obs[0], deterministic=True)
        action_1, _ = human_policy.predict(obs[1], deterministic=True)

        total_action = [action_0, action_1]

        total_action, real_execute_macro_actions = shared_env._computeLowLevelActions(
            total_action
        )

        obs, rewards, dones, info = shared_env.step(total_action)

        # accumulate team reward
        reward_this += float(rewards[0]) + float(rewards[1])

        obs = shared_env._get_macro_obs()

        # optional rendering
        # shared_env.render(mode="rgb_array")

        print("Step:", step, "Reward:", reward_this)

        # stop if done
        if isinstance(dones, (list, tuple, np.ndarray)):
            if any(dones):
                break
        else:
            if dones:
                break

    print("Final reward:", reward_this)

    return reward_this




# Assume the human policy parameter is "no, 0, -10"

human_helping = "no"
human_sp_agent0 = 0
human_sp_agent1 = -10

path_agent_human = build_policy_path(
    agent_id=1,
    helping=human_helping,
    sp_agent0=human_sp_agent0,
    sp_agent1=human_sp_agent1
)

print("Loading Human model from:", path_agent_human)

model_human = PPO.load(path_agent_human, env=env_agent_1)


returned_reward = evaluate(["yes", 0, 0], 200, model_human)
print(returned_reward)