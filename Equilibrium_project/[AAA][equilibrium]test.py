import gym
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import matplotlib.pyplot as plt
import time
import random
from gym_macro_overcooked.items import Tomato, Lettuce, Onion, Plate, Knife, Delivery, Agent, Food, DirtyPlate



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

        # human_agent_previous_location = [self.agent[1].x, self.agent[1].y]

        actions = [0, 0]

        other_agent_action = self.other_agent_model.predict(self.obs[1 - self.agent_index])

        actions[self.agent_index] = action

        actions[1 - self.agent_index] = other_agent_action[0]

        primary_actions, _ = self.env._computeLowLevelActions(actions)

        self.obs, rewards, dones, info = self.env.step(primary_actions)

        self.obs = self.env._get_macro_obs()



        return self.obs[self.agent_index], rewards[2], dones, info




def check_get_lettuce_preference(env, action_up, action_down, boolean_preference_up, boolean_preference_down, action_dones):

    # 如果agent采取了某种action，则单独加分

    reward_bonus_up = 0
    reward_bonus_down = 0

    agent_up = env.agent[0]
    agent_down = env.agent[1]

    knife = env.knife[0]

    if boolean_preference_up and not agent_up.holding and action_dones[0] == False and (env.macroActionName[action_up] == "get lettuce 1" or env.macroActionName[action_up] == "get lettuce 2"):
        reward_bonus_up = 100

    if boolean_preference_down and not agent_down.holding and action_dones[1] == False and (env.macroActionName[action_down] == "get lettuce 1" or env.macroActionName[action_down] == "get lettuce 2"):
        reward_bonus_down = 100

    return reward_bonus_up, reward_bonus_down



def check_get_plate_preference(env, action_up, action_down, boolean_preference_up, boolean_preference_down, action_dones):

    # 如果agent采取了某种action，则单独加分

    reward_bonus_up = 0
    reward_bonus_down = 0

    agent_up = env.agent[0]
    agent_down = env.agent[1]


    if boolean_preference_up and not agent_up.holding and action_dones[0] == False and (env.macroActionName[action_up] == "get plate 1" or env.macroActionName[action_up] == "get plate 2"):
        reward_bonus_up = 100

    if boolean_preference_down and not agent_down.holding and action_dones[1] == False and (env.macroActionName[action_down] == "get plate 1" or env.macroActionName[action_down] == "get plate 2"):
        reward_bonus_down = 100

    return reward_bonus_up, reward_bonus_down



def check_go_to_knife_preference(env, action_up, action_down, boolean_preference_up, boolean_preference_down, action_dones):

    # 如果agent采取了某种action，则单独加分

    reward_bonus_up = 0
    reward_bonus_down = 0

    agent_up = env.agent[0]
    agent_down = env.agent[1]


    if boolean_preference_up and agent_up.holding and action_dones[0] == False and isinstance(agent_up.holding, Lettuce) and (env.macroActionName[action_up] == "go to knife 1" or env.macroActionName[action_up] == "go to knife 2"):
        reward_bonus_up = 100

    if boolean_preference_down and agent_down.holding and action_dones[1] == False and isinstance(agent_down.holding, Lettuce) and (env.macroActionName[action_down] == "go to knife 1" or env.macroActionName[action_down] == "go to knife 2"):
        reward_bonus_down = 100

    return reward_bonus_up, reward_bonus_down



def check_deliver_preference(env, action_up, action_down, boolean_preference_up, boolean_preference_down, action_dones):

    # 如果agent采取了某种action，则单独加分

    reward_bonus_up = 0
    reward_bonus_down = 0

    agent_up = env.agent[0]
    agent_down = env.agent[1]


    if boolean_preference_up and agent_up.holding and action_dones[0] == False and agent_up.holding.containing and (env.macroActionName[action_up] == "deliver"):
        reward_bonus_up = 100

    if boolean_preference_down and agent_down.holding and action_dones[1] == False and agent_down.holding.containing and (env.macroActionName[action_down] == "deliver"):
        reward_bonus_down = 100

    return reward_bonus_up, reward_bonus_down







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



mac_env_id = 'Overcooked-MA-equilibrium-v1'
env_params = {
    'grid_dim': [5, 8],
    'task': ["lettuce salad"],
    'rewardList': rewardList,
    'map_type': "counter",
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



model_agent_0 = PPO.load("../policy_pool_newmap/[equilibrium][counter]agent0_a0sp_0_a1sp_0_helping_False_gamma0.8_0.8/model_500000", env=env_agent_0)
model_agent_1 = PPO.load("../policy_pool_newmap/[equilibrium][counter]agent1_a0sp_0_a1sp_0_helping_False_gamma0.8_0.8/model_500000", env=env_agent_1)



model_agent_0 = PPO.load("../policy_pool_newmap/[equilibrium][counter]agent0_a0sp_0_a1sp_0_helping_True_gamma0.8_0.8/model_500000", env=env_agent_0)
model_agent_1 = PPO.load("../policy_pool_newmap/[equilibrium][counter]agent1_a0sp_0_a1sp_0_helping_True_gamma0.8_0.8/model_500000", env=env_agent_1)


model_agent_0 = PPO.load("../policy_pool_newmap/[equilibrium][counter]agent0_a0sp_0_a1sp_0_helping_True_gamma0.95_0.95/model_500000", env=env_agent_0)
model_agent_1 = PPO.load("../policy_pool_newmap/[equilibrium][counter]agent1_a0sp_0_a1sp_0_helping_True_gamma0.95_0.95/model_500000", env=env_agent_1)


model_agent_0 = PPO.load("final_trained_models/[equilibrium][counter]agent0_a0sp_0_a1sp_0_helping_True_gamma0.8_0.8/model_1000000", env=env_agent_0)
model_agent_1 = PPO.load("final_trained_models/[equilibrium][counter]agent1_a0sp_0_a1sp_0_helping_True_gamma0.8_0.8/model_1000000", env=env_agent_1)



# gamma, reward (helping)


# Test the trained models
obs = shared_env.reset()

import time
import cv2
import numpy as np

reward_this = 0


"""For render to a mp4 video"""

# # ===== Render the first frame=====
frame0 = shared_env.render(mode="rgb_array")  # RGB, HxWx3
h, w, _ = frame0.shape

fps = 10
video_path = "rollout_1st.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

video_writer.write(cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))


for step in range(200):
    action_0, _states_0 = model_agent_0.predict(obs[0])
    action_1, _states_1 = model_agent_1.predict(obs[1])

    total_action = [action_0, action_1]


    # print('agent action: ', shared_env.macroActionName[action_0])


    human_agent_previous_location = [shared_env.agent[1].x, shared_env.agent[1].y]


    total_action, real_execute_macro_actions = shared_env._computeLowLevelActions(
        total_action
    )





    # boolean_get_lettuce_preference_up = False
    # boolean_get_lettuce_preference_down = False


    # boolean_get_plate_preference_up = False
    # boolean_get_plate_preference_down = True


    # boolean_go_to_knife_preference_up = False
    # boolean_go_to_knife_preference_down = False

    
    # boolean_deliver_preference_up = False
    # boolean_deliver_preference_down = False



    # get_lettuce_preference_reward_up, get_lettuce_preference_reward_down = check_get_lettuce_preference(shared_env, real_execute_macro_actions[0], real_execute_macro_actions[1], boolean_get_lettuce_preference_up, boolean_get_lettuce_preference_down, action_dones)

    # get_plate_preference_reward_up, get_plate_preference_reward_down = check_get_plate_preference(shared_env, real_execute_macro_actions[0], real_execute_macro_actions[1], boolean_get_plate_preference_up, boolean_get_plate_preference_down, action_dones)

    # go_to_knife_preference_reward_up, go_to_knife_preference_reward_down = check_go_to_knife_preference(shared_env, real_execute_macro_actions[0], real_execute_macro_actions[1], boolean_go_to_knife_preference_up, boolean_go_to_knife_preference_down, action_dones)

    # deliver_preference_reward_up, deliver_preference_reward_down = check_deliver_preference(shared_env, real_execute_macro_actions[0], real_execute_macro_actions[1], boolean_deliver_preference_up, boolean_deliver_preference_down, action_dones)

    # print('get_lettuce_preference_reward_up: ', get_lettuce_preference_reward_up)
    # print('get_lettuce_preference_reward_down: ', get_lettuce_preference_reward_down)
    # print('get_plate_preference_reward_up: ', get_plate_preference_reward_up)
    # print('get_plate_preference_reward_down: ', get_plate_preference_reward_down)
    # print('go_to_knife_preference_reward_up: ', go_to_knife_preference_reward_up)
    # print('go_to_knife_preference_reward_down: ', go_to_knife_preference_reward_down)
    # print('deliver_preference_reward_up: ', deliver_preference_reward_up)
    # print('deliver_preference_reward_down: ', deliver_preference_reward_down)










    obs, rewards, dones, info = shared_env.step(total_action)
    reward_this += rewards[0] + rewards[1]

    print("---------")
    obs = shared_env._get_macro_obs()


    human_agent_current_location = [shared_env.agent[1].x, shared_env.agent[1].y]


    print(human_agent_previous_location, human_agent_current_location)    
    


    frame = shared_env.render(mode="rgb_array")


    if frame.shape[0] != h or frame.shape[1] != w:
        frame = frame[:h, :w]

    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(reward_this)
    time.sleep(0.1)

    if isinstance(dones, (list, tuple, np.ndarray)):
        if any(dones):
            break
    else:
        if dones:
            break

# ===== release =====
video_writer.release()
print(f"Saved video to: {video_path}")
