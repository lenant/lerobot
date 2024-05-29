import os

import gymnasium as gym
import numpy as np
import pyarrow as pa
from dora import Node
from gymnasium import spaces

FPS = int(os.getenv("FPS", "30"))
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "640"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "480"))

ALOHA_JOINTS = [
    # absolute joint position
    "left_arm_waist",
    "left_arm_shoulder",
    "left_arm_elbow",
    "left_arm_forearm_roll",
    "left_arm_wrist_angle",
    "left_arm_wrist_rotate",
    # normalized gripper position 0: close, 1: open
    "left_arm_gripper",
    # absolute joint position
    "right_arm_waist",
    "right_arm_shoulder",
    "right_arm_elbow",
    "right_arm_forearm_roll",
    "right_arm_wrist_angle",
    "right_arm_wrist_rotate",
    # normalized gripper position 0: close, 1: open
    "right_arm_gripper",
]
ALOHA_ACTIONS = [
    # position and quaternion for end effector
    "left_arm_waist",
    "left_arm_shoulder",
    "left_arm_elbow",
    "left_arm_forearm_roll",
    "left_arm_wrist_angle",
    "left_arm_wrist_rotate",
    # normalized gripper position (0: close, 1: open)
    "left_arm_gripper",
    "right_arm_waist",
    "right_arm_shoulder",
    "right_arm_elbow",
    "right_arm_forearm_roll",
    "right_arm_wrist_angle",
    "right_arm_wrist_rotate",
    # normalized gripper position (0: close, 1: open)
    "right_arm_gripper",
]


class DoraEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        model="aloha",
        observation_width=IMAGE_WIDTH,
        observation_height=IMAGE_HEIGHT,
        cameras_names=None,
        num_joints=None,
        num_actions=None,
    ):
        """Initializes the Dora environment.

        Args:
            model (str): The model to use. Either 'aloha' or 'custom'.
            observation_width (int): The width of the observation image.
            observation_height (int): The height of the observation image.
            cameras_names (list): A list of camera names to use. If not provided, the default is ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'].
            num_joints (int): The number of joints in the model. If not provided, the default is 14 for 'aloha' and 6 for 'fivedof'.
            num_actions (int): The number of actions in the model. If not provided, the default is 14 for 'aloha' and 6 for 'fivedof'.
        """
        super().__init__()

        # Initialize a new node
        self.node = Node() if os.environ.get("DORA_NODE_CONFIG", None) is not None else None
        self.observation = {"pixels": {}, "agent_pos": None}
        self.terminated = False

        self.observation_height = observation_height
        self.observation_width = observation_width

        # Observation space
        if model == "aloha":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "cam_high": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            ),
                            "cam_low": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            ),
                            "cam_left_wrist": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            ),
                            "cam_right_wrist": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(len(ALOHA_JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )
        elif model == "custom":
            pixel_dict = {}
            for camera in cameras_names:
                assert camera.startswith("cam"), "Camera names must start with 'cam'"
                pixel_dict[camera] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.observation_height, self.observation_width, 3),
                    dtype=np.uint8,
                )
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(pixel_dict),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(num_joints,),
                        dtype=np.float64,
                    ),
                }
            )
        else:
            raise ValueError("Model must be either 'aloha' or 'custom'.")

        # Action space
        if model == "aloha":
            self.action_space = spaces.Box(low=-1, high=1, shape=(len(ALOHA_ACTIONS),), dtype=np.float32)
        elif model == "custom":
            self.action_space = spaces.Box(low=-1, high=1, shape=(num_actions,), dtype=np.float32)

    def _get_obs(self):
        while True:
            event = self.node.next(timeout=0.001)

            ## If event is None, the node event stream is closed and we should terminate the env
            if event is None:
                self.terminated = True
                break

            if event["type"] == "INPUT":
                # Map Image input into pixels key within Aloha environment
                if "cam" in event["id"]:
                    self.observation["pixels"][event["id"]] = (
                        event["value"].to_numpy().reshape(self.observation_height, self.observation_width, 3)
                    )
                else:
                    # Map other inputs into the observation dictionary using the event id as key
                    self.observation[event["id"]] = event["value"].to_numpy()

            # If the event is a timeout error break the update loop.
            elif event["type"] == "ERROR":
                break

    def reset(self, seed: int | None = None):
        self.node.send_output("reset")
        self._get_obs()
        self.terminated = False
        info = {}
        return self.observation, info

    def step(self, action: np.ndarray):
        # Send the action to the dataflow as action key.
        self.node.send_output("action", pa.array(action))
        self._get_obs()
        reward = 0
        terminated = truncated = self.terminated
        info = {}
        return self.observation, reward, terminated, truncated, info

    def render(self): ...

    def close(self):
        # Drop the node
        del self.node