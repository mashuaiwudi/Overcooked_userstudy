# First, My Previous Overcooked User Study Codebase

## How to Run This User Study Codebase

Under this path, you will find the code I used in my previous project to run user studies. I built a simple Python server using **Flask**, and the front end was written in **HTML + JavaScript**. The setup is not complicated. The version of Python is 3.11.

## Start the Backend Server

`backend.py` is the server code. Start it with:

```bash
python backend.py
```

## Run the Frontend

Once the server is running, open:

- `UserStudy/frontend.html`

This will start the study interface and begin data collection.

## Notes

Some comments in the code are written in Chinese. If they appear as garbled text on your side, you can safely ignore them.

For now, both the backend and frontend can run locally. If you later want to deploy everything on a real server, I can guide you through that. For the moment, please start by running everything locally so that you can understand and debug the code first.

## Folder and File Overview

- **utils.py**  
  Contains some utility functions. You can ignore this for now.

- **gym_macro_overcooked/**  
  Includes the core code for the environment.

- **userstudy_models/**  
  Contains the trained agent models for four different Overcooked map layouts.  
  There are four agents for each layout, so in total **16 models (4 × 4)**.  
  Once `backend.py` is running, it will load a specific model file based on the layout ID and model ID sent from the front end.  
  This allows users to play Overcooked with one of these agent partners.

## Final Reminder

Please note that the repository I shared with you is a fairly complex project. It contains the full codebase from my previous project and is much more complicated than what you are currently working on. As a result, some parts may be difficult to understand at first. Try to go through it step by step and see what you can make sense of first.



# Second, Our Human-AI Equilibrium Project!

Above is the introduction to the frontend and backend components of the user study from my previous project.  
Next, I will introduce our current **Human–AI Equilibrium** project.

---

## Code Structure

### `Equilibrium_project/`
This directory contains the current codebase of the equilibrium project.

### Please go into this directory. Then, you will see the following:

### `gym_macro_overcooked/`
This folder contains the environment code.  
If you would like to modify the layouts or change the number and positions of vegetables, plates, and other items in the environment, you should make the changes here.

### `final_trained_models/`
This folder stores the models I have trained so far.

---

## How to Run the Code

### Play the Game Manually
`[AAA][equilibrium]play.py` allows you to play Overcooked manually.

Run:
```bash
python [AAA][equilibrium]play.py
```

A game window will pop up.  
- Press **1** or **2** to select different agents.  
- Use the **arrow keys** to control your actions.

---

## Training Scripts

The following scripts are used to train agents:

- `[AAA][equilibrium]train_highlevelaction_addstep_penalty.py`  
- `[AAA][equilibrium]train_highlevelaction_partitial_obs.py`  
- `[AAA][equilibrium]train_highlevelaction.py`  

Each script corresponds to a different training setting or observation configuration.

---

## Testing Trained Models

`[AAA][equilibrium]test.py` is used to load the trained models and evaluate their performance.

Run this script to see how the trained agents behave in the environment.


# Updates
## 20260303: add a human-alone setting
I update the `Equilibrium_project/gym_macro_overcooked/overcooked_equilibrium.py`

So you need to set `'n_agent': 1` and reset the environment when participants move to the human-alone episode.

### Note that you need to check the returned reward! For human-AI setting, the reward is team reward (agent1+ agent 2). But for human-alone setting, be sure to only return the human agent's reward!



## 20260305: return AI reward for BO

### Please see `[AAA][equilibrium]test_show_gaia_how_to_get_AI_reward.py`

First, you need to add two state variables in the `__init__()` and `reset()` functions of `SingleAgentWrapper_accept_keyboard_action`:

`self.firsttime_down_go_to_counter = True`  
`self.firsttime_up_get_counter_lettuce = True`

Then, you need to add several helper functions. See **Line 75–240**.

Next, you need to insert some code **before `env.step()`**, see **Line 243–254**.

Then, you need to insert some code **after `env.step()`**, see **Line 271–294**.

### All the code I newly added has been enclosed within comment blocks. You can search `Gaia, please see here`

    # Gaia, please see here
    # ====================
    The codes you need to add.
    The codes you need to add.
    The codes you need to add.
    The codes you need to add.
    The codes you need to add.
    The codes you need to add.
    # ====================


## 20260309: Add a new map "counter" and upload the corresponding policy pool

I added a new map called "counter", which has a counter in the middle so that two agents can cooperate.

The trained policies are in the `policy_pool_newmap/`

To run on this new map, you need to update `overcooked_equilibrium.py` and `overcooked_MA_equilibrium_counter.py` to their latest versions.

And you need to change 3 places in the code: `mac_env_id` to `Overcooked-MA-equilibrium-v1`, `grid_dim` to `[5, 8]`, `map_type` to `counter`.

```bash
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
```



## 20260310: Change if cooperation_bonus == True to if cooperation_bonus == 'True'


## 20260312: Add a new map "thin path" and upload the corresponding policy pool

The trained policies are in the `policy_pool_thinpath/`

To run on this new map, you need to update `overcooked_equilibrium.py` and `overcooked_MA_equilibrium_thinpath.py` to their latest versions.

And you need to change 3 places in the code: `mac_env_id` to `Overcooked-MA-equilibrium-v2`, `grid_dim` to `[5, 7]`, `map_type` to `thinpath`.

```bash
mac_env_id = 'Overcooked-MA-equilibrium-v2'
env_params = {
    'grid_dim': [5, 7],
    'task': ["lettuce salad"],
    'rewardList': rewardList,
    'map_type': "thinpath",
    'n_agent': 2,
    'obs_radius': 0,
    'mode': "vector",
    'debug': True
}
```


## 20260323: Retrained a new policy pool on the thin path map.

The trained policies are in the `policy_pool_thinpath_new/`

To run on this new map, you need to update `__init__.py`, `overcooked_equilibrium.py` and `overcooked_MA_equilibrium_thinpath_flexible.py` to their latest versions.

And you need to change `mac_env_id` to `Overcooked-MA-equilibrium-v3`

```bash
mac_env_id = 'Overcooked-MA-equilibrium-v3'
```



## 20260324: Added a description of 3 maps: Map_configuration.md

For different map, the cooperation bonus can be different.
In Gaia's original ```backend_new.py```, the cooperation_bonus always use ```check_action_benevolence()``` function. However, different functions should be used for different maps.

Please see the updated ```backend_NEW.py```:
- check_action_benevolence_circle(env, action_up, action_down, firsttime_down_go_to_counter, firsttime_up_get_counter_lettuce)
- check_action_benevolence_counter(env, action_up, action_down, firsttime_down_go_to_counter, firsttime_up_get_counter_lettuce)
- check_action_benevolence_thinpath(env, action_up, action_down, firsttime_down_go_to_counter, firsttime_up_get_counter_lettuce)

Gaia, please change the cooperation_bonus calculation based on the specific map.




## 20260327: Final policy pools on 3 maps

The final policy pools are:

- final_policy_pool_circle
- final_policy_pool_counter
- final_policy_pool_thinpath

-------
### Some python files need update:
1. Update `overcooked_equilibrium.py`
2. Update `overcooked_MA_equilibrium.py`
3. Update `overcooked_MA_equilibrium_counter.py`
4. Update `overcooked_MA_equilibrium_thinpath.py`

### Also, the AI reward part is updated:
5. In backend_NEW.py, update `def parse_policy_id(policy_id: str)`
6. In backend_NEW.py, update `check_action_benevolence_circle()`
7. In backend_NEW.py, update `check_action_benevolence_counter()`
8. In backend_NEW.py, update `check_action_benevolence_thinpath()`

### The latest configuration of 3 maps/policies

| Map       | Policy Dir              | Policy Prefix                         | Step Penalty     | Cooperation Bonus | Gamma              | Training Steps | Map Type  | Grid Dim | mac_env_id                      |
|-----------|------------------------|--------------------------------------|------------------|-------------------|--------------------|----------------|-----------|----------|----------------------------------|
| Circle    | final_policy_pool_circle     | [equilibrium][circle]agent0_         | 0, -1, -3        | 1000               | 0.8, 0.9, 0.95    | 500000         | circle    | [5, 5]   | Overcooked-MA-equilibrium-v0    |
| Counter   | final_policy_pool_counter    | [equilibrium][counter]agent0_        | 0, -1, -3        | 1000              | 0.8, 0.9, 0.95     | 1500000        | counter   | [5, 8]   | Overcooked-MA-equilibrium-v1    |
| Thinpath  | final_policy_pool_thinpath   | [equilibrium][thinpath]agent0_       | 0, -1, -3        | 1000              | 0.8, 0.9, 0.95     | 1500000        | thinpath  | [5, 7]   | Overcooked-MA-equilibrium-v2    |






## 20260422: Train AI policy on `counter` map with coplay

`final_policy_pool_counter_coplay` contains the new policy pool. Use them as usual.



## 20260427: Train AI policy on `counter` map with coplay and flexible item choice - 5M steps

`final_policy_pool_counter_coplay2` contains the new policy pool. The policy model is `model_5000000`

Be sure to update `gym_macro_overcooked/overcooked_MA_equilibrium_counter_flexible.py` and `gym_macro_overcooked/__init__.py`

### And change the `mac_env_id = 'Overcooked-MA-equilibrium-v4'`!!!!!!!!!!!!!!



## 20260429: Train AI policy on `thinpath` and `circle` map with coplay and flexible item choice - 5M steps

`final_policy_pool_thinpath_coplay` and `final_policy_pool_circle_coplay` contains the new policy pool. The policy model is `model_5000000`

Be sure to update `gym_macro_overcooked/overcooked_MA_equilibrium_counter_flexible.py` and `gym_macro_overcooked/__init__.py`

### Change the `mac_env_id = 'Overcooked-MA-equilibrium-v3'` for `thinpath`
### Change the `mac_env_id = 'Overcooked-MA-equilibrium-v5'` for `circle`