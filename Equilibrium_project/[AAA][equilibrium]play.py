import pygame
import gym
from gym.envs.registration import register
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper

import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

# Define keys for actions
# 0: right, 1: down, 2: left, 3: up, 4: still
KEYS_ACTIONS = {
    pygame.K_UP: 3,       # Up
    pygame.K_RIGHT: 0,    # Right
    pygame.K_DOWN: 1,     # Down
    pygame.K_LEFT: 2,     # Left
}

# Define agent selection keys
AGENT_KEYS = {
    pygame.K_1: 0,
    pygame.K_2: 1,
    pygame.K_3: 2
}




from PIL import Image


def main():

    frames = []


    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Overcooked Control")

    env_id = 'Overcooked-equilibrium-v0'


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
        "penalize using bad lettuce": -20,
        "pick up bad lettuce": -100
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
    env = gym.make(env_id, **env_params)
    

    obs = env.reset()

    frame = env.render(mode='rgb_array')  # Capture the initial frame
    frames.append(frame)  # Append the first frame


    # Render the initial state of the environment
    # env.render()

    # Initial agent selection
    selected_agent = 0

    # Main game loop
    running = True

    step = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            

            if event.type == pygame.KEYDOWN:
                # Handle agent selection
                if event.key in AGENT_KEYS:
                    selected_agent = AGENT_KEYS[event.key]
                
                # Handle actions
                if event.key in KEYS_ACTIONS:
                    step += 1
                    action = [4] * env.n_agent  # Default action is 'stay' for all agents
                    action[selected_agent] = KEYS_ACTIONS[event.key]
                    

                    # if step == 1:
                    #     action = [KEYS_ACTIONS[event.key], 0]
                    # if step == 2:
                    #     action = [KEYS_ACTIONS[event.key], 0]
                    # if step == 3:
                    #     action = [KEYS_ACTIONS[event.key], 3]
                    # if step == 4:
                    #     action = [KEYS_ACTIONS[event.key], 3]

                    print(step)
                    print(action)
                    obs, reward, done, info = env.step(action)


                    print(reward)

                    print(env.agent[0].pomap)



                    frame = env.render(mode='rgb_array')  # Get the rendered frame as an array
                    frames.append(frame)



                    """保存截图"""
                    # img = Image.fromarray(frame)
                    # img.save(f"frame_22222.png", dpi=(300, 300))
                    # print("Frame saved!")



        # screen.fill((0, 0, 0))
        pygame.display.flip()



    # time.sleep(0.5)

    # Save the frames as a video

    # output_path = "Low_Integrity.mp4"
    # imageio.mimsave(output_path, frames, fps=2)  # Save the video at 10 frames per second
    # print(f"Video saved to {output_path}")




    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
    # import random
    # print(random.randint(1, 2))

# env = gym.make(env_id, **env_params)