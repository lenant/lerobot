########################################################################################
# Utilities
########################################################################################


import logging
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache

import cv2
import numpy as np
import torch
import tqdm
from termcolor import colored

from lerobot.common.datasets.populate_dataset import add_frame, safe_stop_image_writer
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, set_global_seed
from lerobot.scripts.eval import get_pretrained_policy_path
from enum import Enum

class HighFiveState(Enum):
    HIGH_FIVE_IN_PROCESS = "HIGH_FIVE_IN_PROCESS"
    HIGH_FIVE_DONE = "HIGH_FIVE_DONE"

state = None

def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1/ dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def has_method(_object: object, method_name: str):
    return hasattr(_object, method_name) and callable(getattr(_object, method_name))


def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def init_policy(pretrained_policy_name_or_path, policy_overrides):
    """Instantiate the policy and load fps, device and use_amp from config yaml"""
    pretrained_policy_path = get_pretrained_policy_path(pretrained_policy_name_or_path)
    hydra_cfg = init_hydra_config(pretrained_policy_path / "config.yaml", policy_overrides)
    policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=pretrained_policy_path)

    # Check device is available
    device = get_safe_torch_device(hydra_cfg.device, log=True)
    use_amp = hydra_cfg.use_amp
    policy_fps = hydra_cfg.env.fps

    policy.eval()
    policy.to(device)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)
    return policy, policy_fps, device, use_amp


def warmup_record(
    robot,
    events,
    enable_teloperation,
    warmup_time_s,
    display_cameras,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        display_cameras=display_cameras,
        events=events,
        fps=fps,
        teleoperate=enable_teloperation,
    )


def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_cameras,
    policy,
    device,
    use_amp,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_cameras=display_cameras,
        dataset=dataset,
        events=events,
        policy=policy,
        device=device,
        use_amp=use_amp,
        fps=fps,
        teleoperate=policy is None,
    )


@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_cameras=False,
    dataset=None,
    events=None,
    policy=None,
    device=None,
    use_amp=None,
    fps=None,
):
    dataset = None
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and fps is not None and dataset["fps"] != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    if 'targets' not in globals():
        targets = []
    if 'last_five' not in globals():
        last_five = [None] * 10
        
    high_five_position = [0, 45,  56, -100, 2, -12]
    home_position = [0, 188, 180, -15, 0, -12]
    max_speed_to_high_five = [7, 25, 25, 15, 30, 15]
    max_speed_to_home = max_speed_to_high_five.copy()
    max_speed_to_home[0] = 5
    max_speed_to_home[1] = 30
    # max_speed = (np.array(max_speed) / 2).tolist()
    
    last_target = None
    
    state = HighFiveState.HIGH_FIVE_DONE

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if teleoperate:
            observation, action = robot.teleop_step(record_data=True)
        else:
            observation = robot.capture_observation()

            if policy is not None:
                target = observation.get("observation.target", None)

                targets.append(target)
                last_five.pop(0)
                last_five.append(target)

                # Create variables target_locked and target_lost
                target_locked = all(t is not None for t in last_five)
                target_lost = all(t is None for t in last_five)
                if target_locked and state is None:
                    state = HighFiveState.HIGH_FIVE_IN_PROCESS
                    
                position = robot.follower_arms["main"].read("Present_Position")
                if state == HighFiveState.HIGH_FIVE_IN_PROCESS:
                    if all(abs(p - hfp) <= 4 for p, hfp in zip(position[1:], high_five_position[1:])):
                        state = HighFiveState.HIGH_FIVE_DONE
                        print("High five completed!")
                    else:
                        if target is not None:
                            last_target = target
                        print("position: ", position)
                        target_position = high_five_position.copy()
                        target_position[0] = 45 - (last_target[0] * (90 / 640)) ## 33 and 61 for not wide
                        target_position = calc_move(target_position, max_speed_to_high_five, position)
                        print("NEW high five position: ", target_position)
                        robot.follower_arms["main"].write("Goal_Position", target_position)
                elif state == HighFiveState.HIGH_FIVE_DONE:
                    if all(abs(p - hfp) <= 4 for p, hfp in zip(position[1:], home_position[1:])):
                        state = None
                        print("Got home!")
                    else:
                        print("position: ", [f"{p:.3f}" for p in position], "home position: ", [f"{hp:.3f}" for hp in home_position])
                        target_position = calc_move(home_position, max_speed_to_home, position)
                        print("NEW to home position: ", target_position)
                        robot.follower_arms["main"].write("Goal_Position", target_position)
                    
                # e
                # elif target_lost:
                #     print("position: ", [f"{p:.3f}" for p in position], "home position: ", [f"{hp:.3f}" for hp in home_position])
                #     target_position = calc_move(home_position, max_speed, position)
                #     print("NEW to home position: ", target_position)
                #     robot.follower_arms["main"].write("Goal_Position", target_position)
                    # time.sleep(0.1)
                    # position_tensor = torch.tensor(position, dtype=torch.float32)
                    # action = robot.send_action(position_tensor)
                # pred_action = predict_action(observation, policy, device, use_amp)
                # # Action can eventually be clipped using `max_relative_target`,
                # # so action actually sent is saved in the dataset.
                # action = robot.send_action(position)
                # action = {"action": action}

        if dataset is not None:
            add_frame(dataset, observation, action)

        if display_cameras and not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break

def calc_move(target_position, max_speed, position):
    new_position = position.copy()
    for i in range(len(new_position)):
        if new_position[i] < target_position[i]:
            new_position[i] = min(new_position[i] + max_speed[i], target_position[i])
        else:
            new_position[i] = max(new_position[i] - max_speed[i], target_position[i])
    return new_position


def reset_environment(robot, events, reset_time_s):
    # TODO(rcadene): refactor warmup_record and reset_environment
    # TODO(alibets): allow for teleop during reset
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    timestamp = 0
    start_vencod_t = time.perf_counter()

    # Wait if necessary
    with tqdm.tqdm(total=reset_time_s, desc="Waiting") as pbar:
        while timestamp < reset_time_s:
            time.sleep(1)
            timestamp = time.perf_counter() - start_vencod_t
            pbar.update(1)
            if events["exit_early"]:
                events["exit_early"] = False
                break


def stop_recording(robot, listener, display_cameras):
    robot.disconnect()

    if not is_headless():
        if listener is not None:
            listener.stop()

        if display_cameras:
            cv2.destroyAllWindows()


def sanity_check_dataset_name(repo_id, policy):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy
    if dataset_name.startswith("eval_") == (policy is None):
        raise ValueError(
            f"Your dataset name begins by 'eval_' ({dataset_name}) but no policy is provided ({policy})."
        )
