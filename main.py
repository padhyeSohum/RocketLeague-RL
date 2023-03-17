# RLGym creates training environments, like OpenAI does with OpenAI Gym, to help users train their models

import numpy as np
# Numpy is a package for scientific computing

from rlgym.envs import Match
# Keeps track of the settings and values of various instances of matches/games

from rlgym.utils.action_parsers import DiscreteAction
# Makes things simpler conceptually and computationally # (a button is either being pressed or it isn't, no in between)

# Stable baselines is a tool that makes the implementation of reinforcement learning a lot easier

from stable_baselines3 import PPO
# Importing the Proximal Policy Optimization (PPO) algorithm from Stable Baselines

from stable_baselines3.common.callbacks import CheckpointCallback
# Saves the model after each fixed number of steps

from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
# vec_env - a method of stacking multiple independent environments into a single environment
# VecMonitor - used to record episode reward, length, time, and other data
# VecNormalize - a moving average, normalizing wrapper for a vectorized environment
# VecCheckNan - raises warnings for when a value is NaN

from stable_baselines3.ppo import MlpPolicy
# implements actor critic using a multi-layered perceptron

from rlgym.utils.obs_builders import AdvancedObs
# creates an observation builder for the environment

from rlgym.utils.state_setters import DefaultState
# state at which each match starts (score = 0-0, time = 0:00, etc.)

from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
# creates conditions for which the state will reset

from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
# allows for multiple instances to be running at the same time

from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
# allows to set reward for each event occurring

from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
# gives the agent a reward for its velocity in the direction of the ball

from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
# gives the agent a reward for the ball's velocity in the direction of the opponent's goal

from rlgym.utils.reward_functions import CombinedReward
# creates the overall reward from each individual reward

if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # calculating discount
    agents_per_match = 2  # 1-on-1s and there is an agent for both teams
    num_instances = 1 # number of instances of rocket league (higher value does not necessarily mean faster training)
    target_steps = 1_000_000
    steps = target_steps // (num_instances * agents_per_match)  # making sure the experience counts line up properly
    batch_size = target_steps//10  # getting the batch size down to something more manageable - 100k in this case
    training_interval = 25_000_000
    mmr_save_frequency = 50_000_000

    def exit_save(model):
        model.save("models/exit_save")

    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match( # match object with settings/parameters
            team_size=1, # means that the bot will be training with 1-on-1s
            tick_skip=frame_skip,
            reward_function=CombinedReward(
            (
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
                EventReward(
                    team_goal=100.0,
                    concede=-100.0,
                    shot=5.0,
                    save=30.0,
                    demo=10.0,
                ),
            ),
            (0.1, 1.0, 1.0)),
            # self_play=True,  #in rlgym 1.2 'self_play' is depreciated. Uncomment line if using an earlier version and comment out spawn_opponents
            spawn_opponents=True,  # this means that the agent will be on both teams, rather than our agent vs a rocket league bot
            terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=DefaultState(),  # Resets to kickoff position
            action_parser=DiscreteAction()  # Discrete > Continuous (less training time)
        )

    env = SB3MultipleInstanceEnv(get_match, num_instances)  # Start 1 instances, waiting 60 seconds between each
    env = VecCheckNan(env)                                # Optional
    env = VecMonitor(env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    try:
        model = PPO.load(
            "models/exit_save.zip",
            env,
            device="auto",
            custom_objects={"n_envs": env.num_envs}, #automatically adjusts to users changing instance count, may encounter shaping error otherwise
            # If you need to adjust parameters mid training, you can use the below example as a guide
            #custom_objects={"n_envs": env.num_envs, "n_steps": steps, "batch_size": batch_size, "n_epochs": 10, "learning_rate": 5e-5}
        )
        print("Loaded previous exit save.")
    except:
        print("No saved model found, creating new model.")
        from torch.nn import Tanh
        policy_kwargs = dict(
            activation_fn=Tanh,
            net_arch=[512, 512, dict(pi=[256, 256, 256], vf=[256, 256, 256])],
        )

        model = PPO(
            MlpPolicy,
            env,
            n_epochs=10,                 # PPO calls for multiple epochs
            policy_kwargs=policy_kwargs, # all of the arguments passed to the policy
            learning_rate=5e-5,          # Around this is fairly common for PPO
            ent_coef=0.01,               # From PPO Atari
            vf_coef=1.,                  # From PPO Atari
            gamma=gamma,                 # Gamma as calculated using half-life
            verbose=3,                   # Print out all the info as we're going
            batch_size=batch_size,       # Batch size as high as possible within reason
            n_steps=steps,               # Number of steps to perform before optimizing network
            tensorboard_log="logs",      # `tensorboard --logdir out/logs` in terminal to see graphs
            device="auto"                # Uses GPU if available
        )

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = CheckpointCallback(round(5_000_000 / env.num_envs), save_path="models", name_prefix="rl_model")

    try:
        mmr_model_target_count = model.num_timesteps + mmr_save_frequency
        while True:
            #may need to reset timesteps when you're running a different number of instances than when you saved the model
            model.learn(training_interval, callback=callback, reset_num_timesteps=False) #can ignore callback if training_interval < callback target
            model.save("models/exit_save")
            if model.num_timesteps >= mmr_model_target_count:
                model.save(f"mmr_models/{model.num_timesteps}")
                mmr_model_target_count += mmr_save_frequency

    except KeyboardInterrupt:
        print("Exiting training")

    print("Saving model")
    exit_save(model)
    print("Save complete")

#######################################################################################################
Visit https://github.com/Impossibum/rlgym_quickstart_tutorial_bot for the original version of the code.
#######################################################################################################
