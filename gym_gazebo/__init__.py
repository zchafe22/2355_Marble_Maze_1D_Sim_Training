import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Gazebo
# ----------------------------------------

# cart pole
register(
    id='GazeboCartPole-v0',
    entry_point='gym_gazebo.envs.gazebo_cartpole:GazeboCartPolev0Env',
)

# marble maze
register(
    id='GazeboMarbleMaze-v0',
    entry_point='gym_gazebo.envs.gazebo_marblemaze:GazeboMarbleMazev0Env',
    max_episode_steps=800,
)