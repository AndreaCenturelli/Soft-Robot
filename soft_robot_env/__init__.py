import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='soft_robot-v0',
    entry_point='soft_robot_env.envs:SoftRobotEnvironment',
    max_episode_steps=100,
    reward_threshold=110.0,
    nondeterministic = True,
)