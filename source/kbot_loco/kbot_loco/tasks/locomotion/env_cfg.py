"""Live KBot locomotion configs; historical experiment configs are in env_cfg.old."""

from __future__ import annotations

import math
import os
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as base_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .assets import (
    IMPLICIT_ANKLE_ACTUATOR_CFG,
    IMPLICIT_HIP_PITCH_KNEE_ACTUATOR_CFG,
    IMPLICIT_HIP_ROLL_ACTUATOR_CFG,
    IMPLICIT_HIP_YAW_ACTUATOR_CFG,
    KBOT_CFG,
    KBOT_TOP4_CFG,
)
from . import mdp


_TOP4_SOLE_CENTER_OFFSETS = [
    (0.030000005, 0.036528746, -0.023478652),
    (0.030000015, -0.036528759, -0.023478660),
]
_TOP4_SOLE_TRACK_Y_M = 0.2854 / 2.0
_TOP4_FOOT_LOCAL_OFFSET_REWARD_NAMES = (
    "dense_foot_swing_speed",
    "dense_swing_foot_target_location_exp",
    "foot_lateral_lane_l1",
    "foot_lateral_lane_max_l1",
    "foot_lateral_spacing_l1",
    "foot_signed_lateral_clearance_l1",
    "foot_sole_lateral_lane_max_l1",
    "gait_cycle_plant_water_level",
    "leg_frontal_sole_plane_max_l1",
    "swing_sole_clearance",
)
_TOP4_LATERAL_TARGET_REWARD_NAMES = (
    "dense_foot_swing_speed",
    "dense_swing_foot_target_location_exp",
    "foot_lateral_lane_l1",
    "foot_lateral_lane_max_l1",
    "foot_sole_lateral_lane_max_l1",
)


def _apply_top4_reward_geometry(rewards) -> None:
    """Patch sole-based reward params for the corrected Top4 foot geometry."""
    foot_local_offsets = list(_TOP4_SOLE_CENTER_OFFSETS)

    for reward_name in _TOP4_FOOT_LOCAL_OFFSET_REWARD_NAMES:
        reward_term = getattr(rewards, reward_name, None)
        if reward_term is not None and reward_term.params is not None:
            reward_term.params["foot_local_offsets"] = foot_local_offsets

    for reward_name in _TOP4_LATERAL_TARGET_REWARD_NAMES:
        reward_term = getattr(rewards, reward_name, None)
        if reward_term is not None and reward_term.params is not None:
            reward_term.params["target_left_y"] = _TOP4_SOLE_TRACK_Y_M
            reward_term.params["target_right_y"] = -_TOP4_SOLE_TRACK_Y_M

    rewards.foot_lateral_spacing_l1.params["target_width"] = 2.0 * _TOP4_SOLE_TRACK_Y_M
    rewards.foot_signed_lateral_clearance_l1.params["minimum_width"] = 2.0 * _TOP4_SOLE_TRACK_Y_M


@configclass
class KBotForwardFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Flat-ground forward velocity tracking task for the asymmetric KBot biped."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = KBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 2048
        self.scene.env_spacing = 2.5

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.env_spacing = self.scene.env_spacing
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.policy.gait_phase = ObsTerm(func=mdp.gait_phase, params={"period_s": 1.0})
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.55)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.debug_vis = False

        self.actions.joint_pos.scale = 0.25

        self.events.physics_material.params["static_friction_range"] = (0.9, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.7, 1.0)
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="floating_base_link")
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0)
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="floating_base_link")
        self.events.base_com.params["com_range"] = {"x": (-0.015, 0.015), "y": (-0.025, 0.025), "z": (-0.01, 0.01)}
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.02, 0.02),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
        }
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)

        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.04)
        self.rewards.track_ang_vel_z_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.05)
        self.rewards.alive = RewTerm(func=base_mdp.is_alive, weight=2.0)
        self.rewards.base_height_l2 = RewTerm(func=base_mdp.base_height_l2, weight=-20.0, params={"target_height": 0.78})
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.25
        self.rewards.flat_orientation_l2.weight = -20.0
        self.rewards.dof_torques_l2.weight = -5.0e-5
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.action_rate_l2.weight = -0.08
        self.rewards.dof_pos_limits.weight = -2.0
        self.rewards.feet_air_time = RewTerm(
            func=base_mdp.feet_air_time_positive_biped,
            weight=0.75,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                "command_name": "base_velocity",
                "threshold": 0.45,
            },
        )
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                "floating_base_link",
                "leg0_shell.*",
                "leg1_shell.*",
                "leg2_shell.*",
                "leg3_shell.*",
            ],
        )
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.feet_air_time.weight = 1.75
        self.rewards.alternating_foot_phase = RewTerm(
            func=mdp.alternating_foot_phase_reward,
            weight=0.35,
            params={
                "period_s": 1.0,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
            },
        )
        self.rewards.lateral_velocity_l2 = RewTerm(func=mdp.lateral_velocity_l2, weight=-7.0)
        self.rewards.yaw_rate_l2 = RewTerm(func=mdp.yaw_rate_l2, weight=-7.0)
        self.rewards.root_lateral_tilt_l2 = RewTerm(func=mdp.root_lateral_tilt_l2, weight=-90.0)
        self.rewards.root_lateral_tilt_ema_l2 = RewTerm(
            func=mdp.root_lateral_tilt_ema_l2,
            weight=-450.0,
            params={"tau_s": 1.5},
        )
        self.rewards.world_heading_l2 = RewTerm(func=mdp.world_heading_l2, weight=-32.0)
        self.rewards.backward_velocity_l2 = RewTerm(func=mdp.backward_velocity_l2, weight=-2.0)
        self.rewards.forward_velocity_below_l2 = RewTerm(
            func=mdp.forward_velocity_below_l2, weight=-20.0, params={"minimum_velocity": 0.30}
        )
        self.rewards.foot_lateral_spacing_l1 = RewTerm(
            func=mdp.foot_lateral_spacing_l1,
            weight=-6.0,
            params={"target_width": 0.24, "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"])},
        )
        self.rewards.foot_signed_lateral_clearance_l1 = RewTerm(
            func=mdp.foot_signed_lateral_clearance_l1,
            weight=-20.0,
            params={"minimum_width": 0.16, "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"])},
        )
        self.rewards.foot_lateral_lane_l1 = RewTerm(
            func=mdp.foot_lateral_lane_l1,
            weight=-7.0,
            params={
                "target_left_y": 0.12,
                "target_right_y": -0.12,
                "tolerance": 0.03,
                "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
            },
        )
        self.rewards.foot_lateral_lane_max_l1 = RewTerm(
            func=mdp.foot_lateral_lane_max_l1,
            weight=-5.0,
            params={
                "target_left_y": 0.12,
                "target_right_y": -0.12,
                "tolerance": 0.02,
                "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
            },
        )
        self.rewards.leg_frontal_plane_l1 = RewTerm(
            func=mdp.leg_frontal_plane_l1,
            weight=-7.0,
            params={
                "tolerance": 0.03,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["leg0_shell", "leg0_shell_2", "leg3_shell1", "leg3_shell11", "foot1", "foot3"],
                    preserve_order=True,
                ),
            },
        )
        leg_plane_asset_cfg = SceneEntityCfg(
            "robot",
            body_names=["leg0_shell", "leg0_shell_2", "leg3_shell1", "leg3_shell11", "foot1", "foot3"],
            preserve_order=True,
        )
        self.rewards.left_leg_frontal_plane_l1 = RewTerm(
            func=mdp.leg_frontal_plane_side_l1,
            weight=-2.0,
            params={"side": "left", "tolerance": 0.015, "asset_cfg": leg_plane_asset_cfg},
        )
        self.rewards.right_leg_frontal_plane_l1 = RewTerm(
            func=mdp.leg_frontal_plane_side_l1,
            weight=-2.0,
            params={"side": "right", "tolerance": 0.015, "asset_cfg": leg_plane_asset_cfg},
        )
        self.rewards.max_leg_frontal_plane_l1 = RewTerm(
            func=mdp.leg_frontal_plane_max_l1,
            weight=-8.0,
            params={"tolerance": 0.01, "asset_cfg": leg_plane_asset_cfg},
        )
        self.rewards.foot_sagittal_separation_l1 = RewTerm(
            func=mdp.foot_sagittal_separation_l1,
            weight=-4.0,
            params={
                "target_length": 0.20,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
            },
        )
        self.rewards.swing_foot_overtake_l1 = RewTerm(
            func=mdp.swing_foot_overtake_l1,
            weight=-14.0,
            params={
                "target_length": 0.16,
                "grace_time": 0.10,
                "target_air_time": 0.45,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
            },
        )
        self.rewards.foot_parallel_l2 = RewTerm(
            func=mdp.foot_parallel_l2,
            weight=-1.5,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"])},
        )
        self.rewards.foot_world_parallel_l2 = RewTerm(
            func=mdp.foot_world_parallel_l2,
            weight=0.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"])},
        )
        self.rewards.foot_world_parallel_max_l2 = RewTerm(
            func=mdp.foot_world_parallel_max_l2,
            weight=0.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"])},
        )
        self.rewards.foot_toe_in_l2 = RewTerm(
            func=mdp.foot_toe_in_l2,
            weight=-8.0,
            params={"tolerance": 0.03, "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"])},
        )
        self.rewards.foot_flat_l2 = RewTerm(
            func=mdp.foot_flat_l2,
            weight=-0.35,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"])},
        )
        self.rewards.stance_foot_flat_l2 = RewTerm(
            func=mdp.stance_foot_flat_l2,
            weight=-2.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
            },
        )
        self.rewards.wobble_joint_vel_l2 = RewTerm(
            func=mdp.joint_velocity_l2,
            weight=-0.04,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_yaw.*", ".*hip_roll.*"])},
        )
        self.rewards.hip_roll_yaw_position_l2 = RewTerm(
            func=mdp.joint_position_l2,
            weight=-12.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_roll.*", ".*hip_yaw.*"])},
        )
        self.rewards.hip_roll_yaw_position_ema_l2 = RewTerm(
            func=mdp.joint_position_ema_l2,
            weight=-36.0,
            params={"tau_s": 1.5, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_roll.*", ".*hip_yaw.*"])},
        )
        self.rewards.low_body_l2 = RewTerm(func=mdp.root_height_below_l2, weight=-30.0, params={"minimum_height": 0.45})
        self.rewards.knee_extension_l1 = RewTerm(
            func=mdp.knee_extension_l1,
            weight=-30.0,
            params={"min_bend": 0.50, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*knee.*"])},
        )
        self.rewards.termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=-500.0)

        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["foot1", "foot3"],
            preserve_order=True,
        )
        self.terminations.base_contact = None
        self.terminations.bad_orientation = None
        self.terminations.low_body = None
        self.terminations.locked_knees = None

        self.decimation = 4
        self.episode_length_s = 8.0


@configclass
class KBotForwardFlatEnvCfg_PLAY(KBotForwardFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.add_base_mass = None
        self.events.base_com = None
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.episode_length_s = 60.0


@configclass
class KBotForwardFlatV2EnvCfg(KBotForwardFlatEnvCfg):
    """V2 clean-walk restart with fewer overlapping reward pressures.

    Diagnostics and checkpoint selection are handled outside the reward function.
    The reward still includes explicit long-window hip-roll pressure because the
    v1 failure mode was a persistent hip/box roll bias over several gait cycles.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = KBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.30)
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)

        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.alternating_foot_phase.weight = 0.25
        self.rewards.upright_alive = RewTerm(
            func=mdp.upright_alive,
            weight=8.0,
            params={"minimum_height": 0.70, "max_tilt": 0.35},
        )

        self.rewards.flat_orientation_l2.weight = -15.0
        self.rewards.lateral_velocity_l2.weight = -5.0
        self.rewards.yaw_rate_l2.weight = -5.0
        self.rewards.world_heading_l2.weight = -20.0
        self.rewards.root_lateral_tilt_l2.weight = -70.0
        self.rewards.root_lateral_tilt_ema_l2.weight = -350.0
        self.rewards.root_lateral_tilt_ema_l2.params["tau_s"] = 2.5

        self.rewards.forward_velocity_below_l2.weight = -8.0
        self.rewards.forward_velocity_below_l2.params["minimum_velocity"] = 0.12
        self.rewards.foot_lateral_spacing_l1.weight = 0.0
        self.rewards.foot_signed_lateral_clearance_l1.weight = -8.0
        self.rewards.foot_signed_lateral_clearance_l1.params["minimum_width"] = 0.26
        self.rewards.foot_lateral_lane_l1.weight = 0.0
        self.rewards.foot_lateral_lane_l1.params["tolerance"] = 0.04
        self.rewards.foot_lateral_lane_max_l1.weight = 0.0
        self.rewards.foot_lateral_lane_max_l1.params["tolerance"] = 0.015
        hip_axis_left_y = 0.15835
        hip_axis_right_y = -0.15805
        sole_center_offsets = [
            (0.03, -0.036528655, -0.0194786795),
            (0.03, -0.036528755, -0.0234786545),
        ]
        self.rewards.foot_sole_lateral_lane_max_l1 = RewTerm(
            func=mdp.foot_sole_lateral_lane_max_l1,
            weight=-14.0,
            params={
                "target_left_y": hip_axis_left_y,
                "target_right_y": hip_axis_right_y,
                "tolerance": 0.008,
                "foot_local_offsets": sole_center_offsets,
                "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
            },
        )

        self.rewards.leg_frontal_plane_l1.weight = -4.0
        self.rewards.leg_frontal_plane_l1.params["tolerance"] = 0.04
        self.rewards.left_leg_frontal_plane_l1.weight = -4.0
        self.rewards.right_leg_frontal_plane_l1.weight = -4.0
        self.rewards.max_leg_frontal_plane_l1.weight = -10.0
        self.rewards.leg_frontal_sole_plane_max_l1 = RewTerm(
            func=mdp.leg_frontal_sole_plane_max_l1,
            weight=-14.0,
            params={
                "tolerance": 0.008,
                "foot_local_offsets": sole_center_offsets,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["leg0_shell", "leg0_shell_2", "leg3_shell1", "leg3_shell11", "foot1", "foot3"],
                    preserve_order=True,
                ),
            },
        )

        self.rewards.foot_sagittal_separation_l1.weight = -3.0
        self.rewards.foot_sagittal_separation_l1.params["target_length"] = 0.22
        self.rewards.swing_foot_overtake_l1.weight = -10.0
        self.rewards.swing_foot_overtake_l1.params["target_length"] = 0.18
        self.rewards.foot_parallel_l2.weight = -1.0
        self.rewards.foot_toe_in_l2.weight = -6.0
        self.rewards.foot_flat_l2.weight = -1.0
        self.rewards.stance_foot_flat_l2.func = mdp.stance_foot_flat_l2
        self.rewards.stance_foot_flat_l2.weight = -6.0

        self.rewards.hip_roll_yaw_position_l2.weight = -8.0
        self.rewards.hip_roll_yaw_position_ema_l2.weight = -24.0
        self.rewards.hip_roll_yaw_position_ema_l2.params["tau_s"] = 2.5
        self.rewards.hip_roll_position_ema_5cycle_l2 = RewTerm(
            func=mdp.joint_position_ema_l2,
            weight=-90.0,
            params={"tau_s": 5.0, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_roll.*"])},
        )
        self.rewards.centered_joint_target_position_l2 = RewTerm(
            func=mdp.joint_target_position_l2,
            weight=-0.5,
            params={
                "targets": {
                    "left_hip_pitch_04": 0.62,
                    "right_hip_pitch_04": 0.62,
                    "left_hip_roll_03": 0.02,
                    "right_hip_roll_03": -0.02,
                    "left_hip_yaw_03": 0.0,
                    "right_hip_yaw_03": 0.0,
                    "left_knee_04": 1.20,
                    "right_knee_04": -1.20,
                    "left_ankle_02": -0.65,
                    "right_ankle_02": 0.65,
                },
            },
        )

        self.rewards.base_height_l2.weight = -35.0
        self.rewards.low_body_l2.weight = -120.0
        self.rewards.low_body_l2.params["minimum_height"] = 0.70
        self.rewards.base_height_l2.params["target_height"] = 0.88
        self.rewards.upright_alive.params["minimum_height"] = 0.70
        self.rewards.knee_extension_l1.weight = -18.0
        self.rewards.knee_extension_l1.params["min_bend"] = 0.35

        self.terminations.low_body = None
        self.terminations.bad_orientation = None


@configclass
class KBotForwardFlatV2EnvCfg_PLAY(KBotForwardFlatV2EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.add_base_mass = None
        self.events.base_com = None
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.episode_length_s = 60.0


@configclass
class KBotForwardFlatV2ScratchPoseBootstrapEnvCfg(KBotForwardFlatV2EnvCfg):
    """Scratch bootstrap from a raw-USD settled hand-authored standing pose."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1024
        self.scene.robot.actuators = {
            "hip_pitch_knee": IMPLICIT_HIP_PITCH_KNEE_ACTUATOR_CFG,
            "hip_roll": IMPLICIT_HIP_ROLL_ACTUATOR_CFG,
            "hip_yaw": IMPLICIT_HIP_YAW_ACTUATOR_CFG,
            "ankles": IMPLICIT_ANKLE_ACTUATOR_CFG,
        }
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.8565)
        self.scene.robot.init_state.joint_pos.update(
            {
                "left_hip_pitch_04": 0.2843153178691864,
                "right_hip_pitch_04": -0.2841152250766754,
                "left_hip_roll_03": 0.0017389939166605473,
                "right_hip_roll_03": 0.0019064429216086864,
                "left_hip_yaw_03": 0.0013319215504452586,
                "right_hip_yaw_03": 0.00043546810047701,
                "left_knee_04": 0.5073038935661316,
                "right_knee_04": -0.5059521198272705,
                "left_ankle_02": -0.24602758884429932,
                "right_ankle_02": 0.24722331762313843,
            }
        )
        self.actions.joint_pos.scale = 0.20
        self.episode_length_s = 4.0

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0

        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.reset_base.params["pose_range"] = {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "yaw": (-0.01, 0.01)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.005, 0.005),
            "y": (-0.005, 0.005),
            "z": (-0.005, 0.005),
            "roll": (-0.005, 0.005),
            "pitch": (-0.005, 0.005),
            "yaw": (-0.005, 0.005),
        }
        self.events.reset_robot_joints.params["position_range"] = (0.995, 1.005)

        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.alternating_foot_phase.weight = 0.0
        self.rewards.forward_velocity_below_l2.weight = 0.0
        self.rewards.foot_sagittal_separation_l1.weight = 0.0
        self.rewards.swing_foot_overtake_l1.weight = 0.0
        self.rewards.foot_lateral_spacing_l1.weight = 0.0
        self.rewards.foot_signed_lateral_clearance_l1.weight = 0.0
        self.rewards.foot_lateral_lane_l1.weight = 0.0
        self.rewards.leg_frontal_plane_l1.weight = 0.0
        self.rewards.foot_parallel_l2.weight = 0.0
        self.rewards.foot_toe_in_l2.weight = 0.0
        self.rewards.stance_foot_flat_l2.weight = 0.0

        self.rewards.base_height_l2.weight = -35.0
        self.rewards.base_height_l2.params["target_height"] = 0.856
        self.rewards.low_body_l2.weight = -120.0
        self.rewards.low_body_l2.params["minimum_height"] = 0.55
        self.rewards.upright_alive.weight = 12.0
        self.rewards.upright_alive.params["minimum_height"] = 0.55
        self.rewards.upright_alive.params["max_tilt"] = 0.50
        self.rewards.flat_orientation_l2.weight = -20.0
        self.rewards.root_lateral_tilt_l2.weight = -50.0
        self.rewards.root_lateral_tilt_ema_l2.weight = -120.0
        self.rewards.hip_roll_yaw_position_l2.weight = 0.0
        self.rewards.hip_roll_yaw_position_ema_l2.weight = 0.0
        self.rewards.hip_roll_position_ema_5cycle_l2.weight = 0.0
        self.rewards.knee_extension_l1.weight = 0.0
        self.rewards.stand_joint_position_l2 = RewTerm(
            func=mdp.joint_position_l2,
            weight=-4.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        self.rewards.termination_penalty.weight = -250.0

        self.terminations.low_body = None
        self.terminations.bad_orientation = None


@configclass
class KBotForwardFlatV25ScratchPoseWidthBootstrapEnvCfg(KBotForwardFlatV2ScratchPoseBootstrapEnvCfg):
    """V2.5 posed-start bootstrap that improves support width and starts stepping."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.06, 0.14)
        self.commands.base_velocity.resampling_time_range = (4.0, 4.0)

        self.rewards.alive.weight = 1.0
        self.rewards.upright_alive.weight = 8.0
        self.rewards.upright_alive.params["minimum_height"] = 0.76
        self.rewards.upright_alive.params["max_tilt"] = 0.45
        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.01)
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.forward_velocity_below_l2.weight = -8.0
        self.rewards.forward_velocity_below_l2.params["minimum_velocity"] = 0.06
        self.rewards.world_forward_velocity_below_l2 = RewTerm(
            func=mdp.world_forward_velocity_below_l2,
            weight=-30.0,
            params={"minimum_velocity": 0.04},
        )
        self.rewards.world_forward_velocity_clip = RewTerm(
            func=mdp.world_forward_velocity_clip,
            weight=3.0,
            params={"max_velocity": 0.10},
        )
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params["threshold"] = 0.18
        self.rewards.alternating_foot_phase.weight = 0.12
        self.rewards.foot_sagittal_separation_l1.weight = -1.2
        self.rewards.foot_sagittal_separation_l1.params["target_length"] = 0.08
        self.rewards.swing_foot_overtake_l1.weight = -1.6
        self.rewards.swing_foot_overtake_l1.params["target_length"] = 0.06
        self.rewards.swing_foot_overtake_l1.params["target_air_time"] = 0.16
        self.rewards.swing_foot_overtake_l1.params["grace_time"] = 0.04
        self.rewards.lateral_velocity_l2.weight = -18.0
        self.rewards.yaw_rate_l2.weight = -18.0
        self.rewards.world_heading_l2.weight = -80.0
        self.rewards.foot_flat_l2.weight = -3.0
        self.rewards.low_body_l2.params["minimum_height"] = 0.76

        self.rewards.foot_lateral_spacing_l1.weight = -10.0
        self.rewards.foot_lateral_spacing_l1.params["target_width"] = 0.3164
        self.rewards.foot_signed_lateral_clearance_l1.weight = -12.0
        self.rewards.foot_signed_lateral_clearance_l1.params["minimum_width"] = 0.28
        self.rewards.foot_lateral_lane_l1.weight = -4.0
        self.rewards.foot_lateral_lane_l1.params["target_left_y"] = 0.1582
        self.rewards.foot_lateral_lane_l1.params["target_right_y"] = -0.1582
        self.rewards.foot_lateral_lane_l1.params["tolerance"] = 0.08
        self.rewards.foot_lateral_lane_max_l1.weight = -2.0
        self.rewards.foot_lateral_lane_max_l1.params["target_left_y"] = 0.1582
        self.rewards.foot_lateral_lane_max_l1.params["target_right_y"] = -0.1582
        self.rewards.foot_lateral_lane_max_l1.params["tolerance"] = 0.06

        self.rewards.foot_sole_lateral_lane_max_l1.weight = -48.0
        self.rewards.foot_sole_lateral_lane_max_l1.params["target_left_y"] = 0.1582
        self.rewards.foot_sole_lateral_lane_max_l1.params["target_right_y"] = -0.1582
        self.rewards.foot_sole_lateral_lane_max_l1.params["tolerance"] = 0.008

        self.rewards.centered_joint_target_position_l2.weight = 0.0
        self.rewards.stand_joint_position_l2.weight = -0.75
        self.terminations.low_body = DoneTerm(func=mdp.root_height_below, params={"minimum_height": 0.76})
        self.terminations.bad_orientation = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": 0.75})


@configclass
class KBotForwardFlatV25PoseGaitQuality648CompatEnvCfg(KBotForwardFlatV25ScratchPoseWidthBootstrapEnvCfg):
    """Frozen V2.5 pose-gait stack that produced the model_648 checkpoint.

    This compatibility task intentionally excludes later S4 reward terms such
    as valid-step gates, dense step progress, cadence penalties, and contact
    chatter. Use it to replay or continue the 2026-05-09 model_648 lineage
    without silently changing its training pipeline.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot.spawn.articulation_props = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.16)

        self.rewards.world_forward_velocity_clip.weight = 2.5
        self.rewards.world_forward_velocity_clip.params["max_velocity"] = 0.12
        self.rewards.action_rate_l2.weight = -0.09
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.wobble_joint_vel_l2.weight = -0.04
        self.rewards.forward_velocity_below_l2.weight = -6.0
        self.rewards.forward_velocity_below_l2.params["minimum_velocity"] = 0.07
        self.rewards.world_forward_velocity_below_l2.weight = -24.0
        self.rewards.world_forward_velocity_below_l2.params["minimum_velocity"] = 0.05

        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.22
        self.rewards.alternating_foot_phase.weight = 0.18
        self.rewards.foot_sagittal_separation_l1.weight = -2.0
        self.rewards.foot_sagittal_separation_l1.params["target_length"] = 0.10
        self.rewards.swing_foot_overtake_l1.weight = -3.0
        self.rewards.swing_foot_overtake_l1.params["target_length"] = 0.08
        self.rewards.swing_foot_overtake_l1.params["target_air_time"] = 0.20
        self.rewards.swing_foot_overtake_l1.params["grace_time"] = 0.04

        self.rewards.root_lateral_position_l2 = RewTerm(func=mdp.root_lateral_position_l2, weight=-12.0)
        self.rewards.lateral_velocity_l2.weight = -20.0
        self.rewards.yaw_rate_l2.weight = -20.0
        self.rewards.world_heading_l2.weight = -90.0

        self.rewards.foot_lateral_spacing_l1.weight = -9.0
        self.rewards.foot_signed_lateral_clearance_l1.weight = -12.0
        self.rewards.foot_signed_lateral_clearance_l1.params["minimum_width"] = 0.28
        self.rewards.foot_sole_lateral_lane_max_l1.weight = -44.0

        self.rewards.foot_world_parallel_max_l2.weight = -0.8
        self.rewards.foot_flat_l2.weight = -4.0
        self.rewards.stance_foot_flat_l2.weight = -1.2
        self.rewards.action_rate_l2.weight = -0.09
        self.rewards.stand_joint_position_l2.weight = -0.5


@configclass
class KBotForwardFlatV3HandTuned648EnvCfg(KBotForwardFlatV25PoseGaitQuality648CompatEnvCfg):
    """V3 scratch restart from the frozen model_648 reward topology.

    V3 intentionally starts from the code path that trained the V2.5
    `model_648.pt` seed, but as a fresh iteration-0 policy. It does not inherit
    the later S4 step-gate/cadence/chatter reward terms. The tables below are
    the current hand-tuned V3 surface; keep weights grouped by theme, then
    alphabetized by reward name inside each theme, so changes are easy to
    review and revert.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # V3 tuning block. Weight order is theme first, then A-to-Z inside each
        # theme. Keep theme headings stable so hand tuning stays searchable and
        # diff-friendly.
        #
        # Edit rules:
        # - Existing reward weight only: add/edit the name in the matching
        #   reward_weight_groups theme.
        # - Existing reward function change: add the same name in reward_functions.
        # - Existing reward params: add the same name in reward_params.
        # - Brand-new reward term: add a full RewTerm in new_reward_terms and
        #   its editable weight in the matching reward_weight_groups theme.
        # - If a reward fits multiple themes, choose the one that describes the
        #   main behavior you are tuning, and keep the name alphabetized there.
        # Central V3 gait schedule; timing-derived rewards should stay tied to these targets.
        target_root_speed_mps = 0.375
        target_step_length_m = 0.60
        first_step_fraction = 0.5
        swing_phase_fraction = 0.375
        gait_cycle_toe_off_debounce_s = 0.04
        gait_cycle_plant_phase_fraction = 0.10
        swing_overtake_grace_time_s = 0.04
        target_cycle_hz = target_root_speed_mps / max(target_step_length_m, 1.0e-6)
        target_step_hz = 2.0 * target_cycle_hz
        target_cycle_period_s = 1.0 / max(target_cycle_hz, 1.0e-6)
        target_air_time_s = swing_phase_fraction * target_cycle_period_s
        first_step_length_m = target_step_length_m * first_step_fraction
        target_left_y_m = 0.1582
        target_right_y_m = -0.1582
        sole_center_offsets = [
            (0.03, -0.036528655, -0.0194786795),
            (0.03, -0.036528755, -0.0234786545),
        ]

        self.commands.base_velocity.ranges.lin_vel_x = (target_root_speed_mps, target_root_speed_mps)
        self.observations.policy.gait_phase.params.update(
            {
                "period_s": target_cycle_period_s,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                "start_on_first_toe_off": True,
                "toe_off_debounce_s": gait_cycle_toe_off_debounce_s,
            }
        )
        self.terminations.low_body.params["minimum_height"] = float(os.getenv("KBOT_LOW_BODY_TERMINATION_HEIGHT", "0.65"))

        reward_weight_groups = (
            (
                "action_joint_regularization",
                {
                    "action_rate_l2": -0.09,
                    "centered_joint_target_position_l2": 0.0,
                    "dof_acc_l2": -1.0e-7,
                    "dof_pos_limits": -2.0,
                    "dof_torques_l2": -5.0e-5,
                    "hip_roll_position_ema_5cycle_l2": 0.0,
                    "hip_roll_yaw_position_ema_l2": 0.0,
                    "hip_roll_yaw_position_l2": 0.0,
                    "knee_extension_l1": 0.0,
                    "signed_joint_pair_ema_symmetry_l2": -200.0,
                    "stand_joint_position_l2": -1.0,
                    "wobble_joint_vel_l2": -0.04,
                },
            ),
            (
                "forward_heading_tracking",
                {
                    "backward_velocity_l2": -100.0,
                    "forward_velocity_below_l2": -6.0,
                    "track_ang_vel_z_exp": 1.0,
                    "track_lin_vel_xy_exp": 30.0,
                    "world_forward_velocity_below_l2": -24.0,
                    "world_forward_velocity_clip": 5.0,
                    "world_heading_l2": -1000.0,
                    "yaw_rate_l2": -200.0,
                },
            ),
            (
                "gait_step_timing",
                {
                    "alternating_foot_phase": 0.0,
                    "alternating_step_symmetry_l2": -8.0,
                    "contact_duty_symmetry_l2": -10.0,
                    "dense_foot_swing_speed": 40.0,
                    "dense_swing_foot_target_location_exp": 100.0,
                    "dense_swing_step_length": 0.0,
                    "feet_air_time": 0.0,
                    "foot_sagittal_separation_l1": -12.0,
                    "foot_retreat": -12.0,
                    "gait_cycle_plant_water_level": 1.0,
                    "gait_cycle_support": 1.0,
                    "swing_foot_overtake_l1": -150.0,
                    "swing_sole_clearance": 1.0,
                },
            ),
            (
                "lateral_centerline_width",
                {
                    "foot_lateral_lane_l1": -4.0,
                    "foot_lateral_lane_max_l1": -2.0,
                    "foot_lateral_spacing_l1": -30.0,
                    "foot_signed_lateral_clearance_l1": -24.0,
                    "foot_sole_lateral_lane_max_l1": -100.0,
                    "lateral_away_from_center_l2": -600.0,
                    "lateral_velocity_l2": -20.0,
                    "root_lateral_position_l2": -500.0,
                },
            ),
            (
                "leg_frontal_plane",
                {
                    "left_leg_frontal_plane_l1": -4.0,
                    "leg_frontal_plane_l1": 0.0,
                    "leg_frontal_sole_plane_max_l1": -14.0,
                    "max_leg_frontal_plane_l1": -10.0,
                    "right_leg_frontal_plane_l1": -4.0,
                },
            ),
            (
                "posture_survival",
                {
                    "alive": 1.0,
                    "ang_vel_xy_l2": -0.25,
                    "base_height_l2": -35.0,
                    "flat_orientation_l2": -20.0,
                    "lin_vel_z_l2": -2.0,
                    "low_body_l2": -120.0,
                    "root_lateral_tilt_ema_l2": -12000.0,
                    "root_lateral_tilt_l2": -90.0,
                    "termination_penalty": -3000.0,
                    "undesired_contacts": -2.0,
                    "upright_alive": 8.0,
                },
            ),
            (
                "sole_foot_orientation",
                {
                    "foot_flat_l2": 0.0,
                    "foot_parallel_l2": -30.0,
                    "foot_toe_in_l2": 0.0,
                    "foot_world_parallel_l2": -10.0,
                    "foot_world_parallel_max_l2": -10.0,
                    "stance_foot_flat_l2": -1.2,
                },
            ),
        )

        reward_weights = {}
        for _theme, weights in reward_weight_groups:
            duplicate_names = reward_weights.keys() & weights.keys()
            if duplicate_names:
                raise ValueError(f"Duplicate V3 reward weights: {sorted(duplicate_names)}")
            reward_weights.update(weights)

        reward_functions = {
            "track_lin_vel_xy_exp": mdp.upright_centerline_heading_gated_track_lin_vel_xy_exp,
            "world_forward_velocity_clip": mdp.upright_centerline_heading_gated_world_forward_velocity_clip,
        }

        adaptive_cycle_ema_params = {
            "cycle_duration_smoothing_cycles": 5.0,
            "ema_cycle_count": 5.0,
            "max_cycle_duration_s": 2.0,
            "max_tau_s": 10.0,
            "min_cycle_duration_s": 0.25,
            "min_tau_s": 0.75,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
        }

        reward_params = {
            "alternating_foot_phase": {
                "period_s": target_cycle_period_s,
            },
            "dense_foot_swing_speed": {
                "foot_local_offsets": sole_center_offsets,
                "max_step_credit": 1.5,
                "max_height": 0.020,
                "max_tilt": 0.45,
                "min_height": 0.005,
                "minimum_height": 0.76,
                "target_air_time": target_air_time_s,
                "target_left_y": target_left_y_m,
                "target_length": target_step_length_m,
                "target_right_y": target_right_y_m,
                "y_linear_radius": 0.12,
                "y_scale": 0.04,
                "z_scale": 0.005,
            },
            "dense_swing_step_length": {
                "crossover_fraction": 0.66,
                "lambda_per_m": 0.0,
                "linear_gain": 0.0,
                "target_length": target_step_length_m,
            },
            "dense_swing_foot_target_location_exp": {
                "first_target_fraction": first_step_fraction,
                "linear_progress_scale": 0.2,
                "max_tilt": 0.45,
                "minimum_height": 0.76,
                "period_s": target_cycle_period_s,
                "plant_phase_fraction": gait_cycle_plant_phase_fraction,
                "smooth_max_lambda": 0.2,
                "swing_phase_fraction": swing_phase_fraction,
                "target_left_y": target_left_y_m,
                "target_length": target_step_length_m,
                "target_right_y": target_right_y_m,
                "toe_off_debounce_s": gait_cycle_toe_off_debounce_s,
                "x_scale": 0.15,
                "y_scale": 0.08,
            },
            "feet_air_time": {
                "threshold": target_air_time_s,
            },
            "foot_sagittal_separation_l1": {
                "first_target_fraction": first_step_fraction,
                "target_length": target_step_length_m,
            },
            "forward_velocity_below_l2": {
                "minimum_velocity": 0.07,
            },
            "hip_roll_position_ema_5cycle_l2": adaptive_cycle_ema_params,
            "hip_roll_yaw_position_ema_l2": adaptive_cycle_ema_params,
            "root_lateral_tilt_ema_l2": adaptive_cycle_ema_params,
            "foot_retreat": {
                "retreat_epsilon": 0.002,
            },
            "gait_cycle_support": {
                "airborne_penalty": 1.0,
                "period_s": target_cycle_period_s,
                "plant_phase_fraction": gait_cycle_plant_phase_fraction,
                "precycle_airborne_penalty": 1.0,
                "precycle_double_support_reward": 0.0,
                "precycle_single_support_reward": 0.0,
                "shift_single_support_reward": -1.0,
                "swing_double_support_reward": -1.0,
                "swing_phase_fraction": swing_phase_fraction,
                "toe_off_debounce_s": gait_cycle_toe_off_debounce_s,
                "wrong_single_penalty": 1.0,
            },
            "gait_cycle_plant_water_level": {
                "extra_swing_takeoff_penalty": 1.0,
                "foot_local_offsets": sole_center_offsets,
                "max_tilt": 0.45,
                "minimum_height": 0.76,
                "minimum_water_level": 0.002,
                "outside_plant_touchdown_penalty": 1.0,
                "period_s": target_cycle_period_s,
                "plant_phase_fraction": gait_cycle_plant_phase_fraction,
                "post_plant_lift_penalty": 1.0,
                "post_plant_up_penalty": 1.0,
                "retreat_epsilon": 0.002,
                "retreat_penalty": 1.0,
                "retreat_scale": 0.020,
                "swing_phase_fraction": swing_phase_fraction,
                "toe_off_debounce_s": gait_cycle_toe_off_debounce_s,
                "up_penalty": 1.0,
                "water_level": 0.010,
                "z_epsilon": 0.001,
            },
            "swing_foot_overtake_l1": {
                "grace_time": swing_overtake_grace_time_s,
                "target_air_time": target_air_time_s,
                "target_length": first_step_length_m,
            },
            "track_lin_vel_xy_exp": {
                "centerline_target_y": 0.0,
                "centerline_width_sq": 0.01,
                "heading_width_sq": 0.01,
                "max_tilt": 0.45,
                "minimum_height": 0.76,
            },
            "world_forward_velocity_below_l2": {
                "minimum_velocity": 0.05,
            },
            "world_forward_velocity_clip": {
                "centerline_target_y": 0.0,
                "centerline_width_sq": 0.01,
                "heading_width_sq": 0.01,
                "max_tilt": 0.45,
                "max_velocity": target_root_speed_mps,
                "minimum_height": 0.76,
            },
        }

        new_reward_terms = {
            "alternating_step_symmetry_l2": RewTerm(
                func=mdp.alternating_step_symmetry_l2,
                weight=reward_weights["alternating_step_symmetry_l2"],
                params={
                    "advance_scale": 0.08,
                    "duration_scale": 0.20,
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                    "tau_s": 1.0,
                    **adaptive_cycle_ema_params,
                },
            ),
            "contact_duty_symmetry_l2": RewTerm(
                func=mdp.contact_duty_symmetry_l2,
                weight=reward_weights["contact_duty_symmetry_l2"],
                params={
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                    "tau_s": 1.0,
                    **adaptive_cycle_ema_params,
                },
            ),
            "dense_swing_step_length": RewTerm(
                func=mdp.dense_swing_step_length_reward,
                weight=reward_weights["dense_swing_step_length"],
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
                    "command_name": "base_velocity",
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                },
            ),
            "dense_foot_swing_speed": RewTerm(
                func=mdp.dense_foot_swing_speed_reward,
                weight=reward_weights["dense_foot_swing_speed"],
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
                    "command_name": "base_velocity",
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                },
            ),
            "dense_swing_foot_target_location_exp": RewTerm(
                func=mdp.dense_swing_foot_target_location_exp,
                weight=reward_weights["dense_swing_foot_target_location_exp"],
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
                    "command_name": "base_velocity",
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                },
            ),
            "lateral_away_from_center_l2": RewTerm(
                func=mdp.lateral_away_from_center_l2,
                weight=reward_weights["lateral_away_from_center_l2"],
            ),
            "signed_joint_pair_ema_symmetry_l2": RewTerm(
                func=mdp.signed_joint_pair_ema_symmetry_l2,
                weight=reward_weights["signed_joint_pair_ema_symmetry_l2"],
                params={
                    "joint_pairs": [
                        ("left_hip_pitch_04", "right_hip_pitch_04", -1.0),
                        ("left_hip_roll_03", "right_hip_roll_03", -1.0),
                        ("left_hip_yaw_03", "right_hip_yaw_03", -1.0),
                        ("left_knee_04", "right_knee_04", -1.0),
                        ("left_ankle_02", "right_ankle_02", -1.0),
                    ],
                    "tau_s": 1.0,
                    **adaptive_cycle_ema_params,
                },
            ),
            "foot_retreat": RewTerm(
                func=mdp.foot_retreat,
                weight=reward_weights["foot_retreat"],
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
                    "command_name": "base_velocity",
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                },
            ),
            "gait_cycle_support": RewTerm(
                func=mdp.gait_cycle_support_reward,
                weight=reward_weights["gait_cycle_support"],
                params={
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                },
            ),
            "gait_cycle_plant_water_level": RewTerm(
                func=mdp.gait_cycle_plant_water_level_reward,
                weight=reward_weights["gait_cycle_plant_water_level"],
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
                    "command_name": "base_velocity",
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                },
            ),
            "swing_sole_clearance": RewTerm(
                func=mdp.swing_sole_clearance_reward,
                weight=reward_weights["swing_sole_clearance"],
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
                    "command_name": "base_velocity",
                    "drag_floor": 0.002,
                    "drag_weight": 3.0,
                    "foot_local_offsets": sole_center_offsets,
                    "max_tilt": 0.45,
                    "minimum_height": 0.76,
                    "over_height": 0.020,
                    "over_penalty_cap": 4.0,
                    "over_scale": 0.010,
                    "over_weight": 1.0,
                    "period_s": target_cycle_period_s,
                    "plant_phase_fraction": gait_cycle_plant_phase_fraction,
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                    "swing_phase_fraction": swing_phase_fraction,
                    "target_height": 0.010,
                    "toe_off_debounce_s": gait_cycle_toe_off_debounce_s,
                },
            ),
        }

        for reward_name, reward_term in new_reward_terms.items():
            setattr(self.rewards, reward_name, reward_term)
        for reward_name, reward_func in reward_functions.items():
            getattr(self.rewards, reward_name).func = reward_func
        for reward_name, params in reward_params.items():
            reward_term = getattr(self.rewards, reward_name)
            if reward_term.params is None:
                reward_term.params = {}
            reward_term.params.update(params)
        for reward_name, weight in reward_weights.items():
            getattr(self.rewards, reward_name).weight = weight


@configclass
class KBotForwardFlatV31ScratchEnvCfg(KBotForwardFlatV3HandTuned648EnvCfg):
    """V3.1 scratch policy: current cyclic V3 rewards with May 31-like weights."""

    def __post_init__(self) -> None:
        super().__post_init__()

        reward_weight_groups = (
            (
                "action_joint_regularization",
                {
                    "action_rate_l2": -0.09,
                    "centered_joint_target_position_l2": 0.0,
                    "dof_acc_l2": -1.0e-7,
                    "dof_pos_limits": -2.0,
                    "dof_torques_l2": -5.0e-5,
                    "hip_roll_position_ema_5cycle_l2": 0.0,
                    "hip_roll_yaw_position_ema_l2": 0.0,
                    "hip_roll_yaw_position_l2": 0.0,
                    "knee_extension_l1": 0.0,
                    "signed_joint_pair_ema_symmetry_l2": -3.0,
                    "stand_joint_position_l2": -0.5,
                    "wobble_joint_vel_l2": -0.04,
                },
            ),
            (
                "forward_heading_tracking",
                {
                    "backward_velocity_l2": -2.0,
                    "forward_velocity_below_l2": -6.0,
                    "track_ang_vel_z_exp": 1.0,
                    "track_lin_vel_xy_exp": 30.0,
                    "world_forward_velocity_below_l2": -24.0,
                    "world_forward_velocity_clip": 30.0,
                    "world_heading_l2": -90.0,
                    "yaw_rate_l2": -20.0,
                },
            ),
            (
                "gait_step_timing",
                {
                    "alternating_foot_phase": 0.0,
                    "alternating_step_symmetry_l2": -2.0,
                    "contact_duty_symmetry_l2": -2.0,
                    "dense_foot_swing_speed": 20.0,
                    "dense_swing_foot_target_location_exp": 40.0,
                    "dense_swing_step_length": 0.0,
                    "feet_air_time": 0.0,
                    "foot_sagittal_separation_l1": -4.0,
                    "foot_retreat": -6.0,
                    "gait_cycle_plant_water_level": 1.0,
                    "gait_cycle_support": 1.0,
                    "swing_foot_overtake_l1": -80.0,
                    "swing_sole_clearance": 1.0,
                },
            ),
            (
                "lateral_centerline_width",
                {
                    "foot_lateral_lane_l1": -4.0,
                    "foot_lateral_lane_max_l1": -2.0,
                    "foot_lateral_spacing_l1": -9.0,
                    "foot_signed_lateral_clearance_l1": -12.0,
                    "foot_sole_lateral_lane_max_l1": -44.0,
                    "lateral_away_from_center_l2": 0.0,
                    "lateral_velocity_l2": -20.0,
                    "root_lateral_position_l2": -12.0,
                },
            ),
            (
                "leg_frontal_plane",
                {
                    "left_leg_frontal_plane_l1": -4.0,
                    "leg_frontal_plane_l1": 0.0,
                    "leg_frontal_sole_plane_max_l1": -14.0,
                    "max_leg_frontal_plane_l1": -10.0,
                    "right_leg_frontal_plane_l1": -4.0,
                },
            ),
            (
                "posture_survival",
                {
                    "alive": 1.0,
                    "ang_vel_xy_l2": -0.25,
                    "base_height_l2": -35.0,
                    "flat_orientation_l2": -20.0,
                    "lin_vel_z_l2": -2.0,
                    "low_body_l2": -120.0,
                    "root_lateral_tilt_ema_l2": -120.0,
                    "root_lateral_tilt_l2": -50.0,
                    "termination_penalty": -750.0,
                    "undesired_contacts": -2.0,
                    "upright_alive": 8.0,
                },
            ),
            (
                "sole_foot_orientation",
                {
                    "foot_flat_l2": 0.0,
                    "foot_parallel_l2": 0.0,
                    "foot_toe_in_l2": 0.0,
                    "foot_world_parallel_l2": 0.0,
                    "foot_world_parallel_max_l2": -0.8,
                    "stance_foot_flat_l2": -1.2,
                },
            ),
        )

        reward_weights = {}
        for _theme, weights in reward_weight_groups:
            duplicate_names = reward_weights.keys() & weights.keys()
            if duplicate_names:
                raise ValueError(f"Duplicate V3.1 reward weights: {sorted(duplicate_names)}")
            reward_weights.update(weights)

        for reward_name, weight in reward_weights.items():
            getattr(self.rewards, reward_name).weight = weight


@configclass
class KBotForwardFlatV4Top4StarterEnvCfg(KBotForwardFlatV31ScratchEnvCfg):
    """V4 task: corrected Top4 robot with conservative gait-quality rewards."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.375, 0.375)
        self.decimation = 2
        self.sim.render_interval = self.decimation

        self.scene.robot = KBOT_TOP4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators = {
            "hip_pitch_knee": IMPLICIT_HIP_PITCH_KNEE_ACTUATOR_CFG,
            "hip_roll": IMPLICIT_HIP_ROLL_ACTUATOR_CFG,
            "hip_yaw": IMPLICIT_HIP_YAW_ACTUATOR_CFG,
            "ankles": IMPLICIT_ANKLE_ACTUATOR_CFG,
        }
        self.scene.robot.spawn.articulation_props = None
        self.scene.num_envs = 4096

        _apply_top4_reward_geometry(self.rewards)
        target_step_duration_s = 0.8

        # V4 tuning block. Keep theme headings stable and reward names
        # alphabetized inside each theme; put function and param edits beside
        # the table instead of mixing them into the weight assignments.
        reward_weight_groups = (
            (
                "action_joint_regularization",
                {
                    "action_rate_l2": -0.09,
                    "centered_joint_target_position_l2": 0.0,
                    "dof_acc_l2": -1.0e-7,
                    "dof_pos_limits": -2.0,
                    "dof_torques_l2": -5.0e-5,
                    "hip_roll_position_ema_5cycle_l2": 0.0,
                    "hip_roll_yaw_position_ema_l2": 0.0,
                    "hip_roll_yaw_position_l2": 0.0,
                    "knee_extension_l1": 0.0,
                    "signed_joint_pair_ema_symmetry_l2": -3.0,
                    "stand_joint_position_l2": -0.5,
                    "wobble_joint_vel_l2": -0.04,
                },
            ),
            (
                "forward_heading_tracking",
                {
                    "backward_velocity_l2": -2.0,
                    "forward_velocity_below_l2": -6.0,
                    "track_ang_vel_z_exp": 1.0,
                    "track_lin_vel_xy_exp": 30.0,
                    "world_forward_velocity_below_l2": -24.0,
                    "world_forward_velocity_clip": 15.0,
                    "world_heading_l2": -90.0,
                    "yaw_rate_l2": -20.0,
                },
            ),
            (
                "gait_step_timing",
                {
                    "alternating_step_duration_ema_l1": -10,
                    "alternating_step_symmetry_l2": -0.2,
                    "contact_duty_symmetry_l2": -2.0,
                    "dense_foot_swing_speed": 10.0,
                    "dense_swing_foot_target_location_exp": 125.0,
                    "dense_swing_step_length": 0.0,
                    "feet_air_time": 1.0,
                    "foot_retreat": -1.0,
                    "foot_sagittal_separation_l1": -4.0,
                    "gait_cycle_plant_water_level": 2.5,
                    "gait_cycle_support": 20,
                    "swing_foot_overtake_l1": -100.0,
                    "swing_sole_clearance": 75.0,
                },
            ),
            (
                "lateral_centerline_width",
                {
                    "foot_lateral_lane_l1": -4.0,
                    "foot_lateral_lane_max_l1": -2.0,
                    "foot_lateral_spacing_l1": -9.0,
                    "foot_signed_lateral_clearance_l1": -12.0,
                    "foot_sole_lateral_lane_max_l1": -44.0,
                    "lateral_away_from_center_l2": 0.0,
                    "lateral_velocity_l2": -20.0,
                    "root_lateral_position_l2": -12.0,
                },
            ),
            (
                "leg_frontal_plane",
                {
                    "left_leg_frontal_plane_l1": -4.0,
                    "leg_frontal_plane_l1": 0.0,
                    "leg_frontal_sole_plane_max_l1": -14.0,
                    "max_leg_frontal_plane_l1": -10.0,
                    "right_leg_frontal_plane_l1": -4.0,
                },
            ),
            (
                "posture_survival",
                {
                    "alive": 1.0,
                    "ang_vel_xy_l2": -0.25,
                    "base_height_l2": -35.0,
                    "flat_orientation_l2": -20.0,
                    "lin_vel_z_l2": -2.0,
                    "low_body_l2": -120.0,
                    "root_lateral_tilt_ema_l2": -120.0,
                    "root_lateral_tilt_l2": -50.0,
                    "termination_penalty": -750.0,
                    "undesired_contacts": -2.0,
                    "upright_alive": 8.0,
                },
            ),
            (
                "sole_foot_orientation",
                {
                    "foot_flat_l2": 0.0,
                    "foot_parallel_l2": 0.0,
                    "foot_toe_in_l2": 0.0,
                    "foot_world_parallel_l2": 0.0,
                    "foot_world_parallel_max_l2": -0.8,
                    "stance_foot_flat_l2": -1.2,
                },
            ),
        )

        reward_params = {
            "feet_air_time": {
                "threshold": 0.22,
            },
            "swing_foot_overtake_l1": {
                "target_length": 0.30,
            },
            "world_forward_velocity_clip": {
                "max_velocity": 0.375,
            },
        }

        reward_weights = {}
        for _theme, weights in reward_weight_groups:
            duplicate_names = reward_weights.keys() & weights.keys()
            if duplicate_names:
                raise ValueError(f"Duplicate V4 reward weights: {sorted(duplicate_names)}")
            reward_weights.update(weights)

        new_reward_terms = {
            "alternating_step_duration_ema_l1": RewTerm(
                func=mdp.alternating_step_duration_ema_l1,
                weight=reward_weights["alternating_step_duration_ema_l1"],
                params={
                    "command_name": "base_velocity",
                    "max_duration_s": 2.0,
                    "min_duration_s": 0.05,
                    "minimum_command_speed": 0.05,
                    "penalty_cap": 1.0,
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                    "smoothing_events": 5.0,
                    "target_duration_s": target_step_duration_s,
                },
            ),
        }

        for reward_name, reward_term in new_reward_terms.items():
            setattr(self.rewards, reward_name, reward_term)

        for reward_name, params in reward_params.items():
            reward_term = getattr(self.rewards, reward_name)
            if reward_term.params is None:
                reward_term.params = {}
            reward_term.params.update(params)

        for reward_name, weight in reward_weights.items():
            getattr(self.rewards, reward_name).weight = weight

        self.rewards.alternating_foot_phase = None


@configclass
class KBotForwardFlatV32May31Top4EnvCfg(KBotForwardFlatV4Top4StarterEnvCfg):
    """V3.2 compatibility task: May 31 0-200 rewards on the corrected Top4 robot."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.decimation = 4
        self.sim.render_interval = self.decimation

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (4.0, 4.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.75, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.gait_phase = ObsTerm(func=mdp.gait_phase, params={"period_s": 1.0})

        self.terminations.low_body.params["minimum_height"] = 0.76
        self.terminations.bad_orientation.params["limit_angle"] = 0.75

        self.rewards.alternating_foot_phase = RewTerm(
            func=mdp.alternating_foot_phase_reward,
            weight=0.18,
            params={
                "period_s": 1.0,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
            },
        )

        reward_weight_groups = (
            (
                "action_joint_regularization",
                {
                    "action_rate_l2": -0.09,
                    "centered_joint_target_position_l2": 0.0,
                    "dof_acc_l2": -1.0e-7,
                    "dof_pos_limits": -2.0,
                    "dof_torques_l2": -5.0e-5,
                    "hip_roll_position_ema_5cycle_l2": 0.0,
                    "hip_roll_yaw_position_ema_l2": 0.0,
                    "hip_roll_yaw_position_l2": 0.0,
                    "knee_extension_l1": 0.0,
                    "signed_joint_pair_ema_symmetry_l2": 0.0,
                    "stand_joint_position_l2": -0.5,
                    "wobble_joint_vel_l2": -0.04,
                },
            ),
            (
                "forward_heading_tracking",
                {
                    "backward_velocity_l2": -2.0,
                    "forward_velocity_below_l2": -6.0,
                    "track_ang_vel_z_exp": 1.0,
                    "track_lin_vel_xy_exp": 30.0,
                    "world_forward_velocity_below_l2": -24.0,
                    "world_forward_velocity_clip": 30.0,
                    "world_heading_l2": -90.0,
                    "yaw_rate_l2": -20.0,
                },
            ),
            (
                "gait_step_timing",
                {
                    "alternating_foot_phase": 0.18,
                    "alternating_step_duration_ema_l1": 0.0,
                    "alternating_step_symmetry_l2": 0.0,
                    "contact_duty_symmetry_l2": 0.0,
                    "dense_foot_swing_speed": 0.0,
                    "dense_swing_foot_target_location_exp": 0.0,
                    "dense_swing_step_length": 0.0,
                    "feet_air_time": 1.0,
                    "foot_retreat": 0.0,
                    "foot_sagittal_separation_l1": -2.0,
                    "gait_cycle_plant_water_level": 0.0,
                    "gait_cycle_support": 0.0,
                    "swing_foot_overtake_l1": -3.0,
                    "swing_sole_clearance": 0.0,
                },
            ),
            (
                "lateral_centerline_width",
                {
                    "foot_lateral_lane_l1": -4.0,
                    "foot_lateral_lane_max_l1": -2.0,
                    "foot_lateral_spacing_l1": -9.0,
                    "foot_signed_lateral_clearance_l1": -12.0,
                    "foot_sole_lateral_lane_max_l1": -44.0,
                    "lateral_away_from_center_l2": 0.0,
                    "lateral_velocity_l2": -20.0,
                    "root_lateral_position_l2": -12.0,
                },
            ),
            (
                "leg_frontal_plane",
                {
                    "left_leg_frontal_plane_l1": -4.0,
                    "leg_frontal_plane_l1": 0.0,
                    "leg_frontal_sole_plane_max_l1": -14.0,
                    "max_leg_frontal_plane_l1": -10.0,
                    "right_leg_frontal_plane_l1": -4.0,
                },
            ),
            (
                "posture_survival",
                {
                    "alive": 1.0,
                    "ang_vel_xy_l2": -0.25,
                    "base_height_l2": -35.0,
                    "flat_orientation_l2": -20.0,
                    "lin_vel_z_l2": -2.0,
                    "low_body_l2": -120.0,
                    "root_lateral_tilt_ema_l2": -120.0,
                    "root_lateral_tilt_l2": -50.0,
                    "termination_penalty": -750.0,
                    "undesired_contacts": -2.0,
                    "upright_alive": 8.0,
                },
            ),
            (
                "sole_foot_orientation",
                {
                    "foot_flat_l2": 0.0,
                    "foot_parallel_l2": 0.0,
                    "foot_toe_in_l2": 0.0,
                    "foot_world_parallel_l2": 0.0,
                    "foot_world_parallel_max_l2": -0.8,
                    "stance_foot_flat_l2": -1.2,
                },
            ),
        )

        reward_weights = {}
        for _theme, weights in reward_weight_groups:
            duplicate_names = reward_weights.keys() & weights.keys()
            if duplicate_names:
                raise ValueError(f"Duplicate V3.2 reward weights: {sorted(duplicate_names)}")
            reward_weights.update(weights)

        reward_functions = {
            "track_lin_vel_xy_exp": mdp.upright_gated_track_lin_vel_xy_exp,
            "world_forward_velocity_clip": mdp.upright_gated_world_forward_velocity_clip,
        }

        reward_params = {
            "feet_air_time": {
                "threshold": 0.22,
            },
            "foot_sagittal_separation_l1": {
                "target_length": 0.60,
            },
            "forward_velocity_below_l2": {
                "minimum_velocity": 0.07,
            },
            "low_body_l2": {
                "minimum_height": 0.76,
            },
            "swing_foot_overtake_l1": {
                "grace_time": 0.04,
                "target_air_time": 0.20,
                "target_length": 0.08,
            },
            "track_lin_vel_xy_exp": {
                "command_name": "base_velocity",
                "max_tilt": 0.45,
                "minimum_height": 0.76,
                "std": 0.1,
            },
            "upright_alive": {
                "max_tilt": 0.45,
                "minimum_height": 0.76,
            },
            "world_forward_velocity_below_l2": {
                "minimum_velocity": 0.05,
            },
            "world_forward_velocity_clip": {
                "max_tilt": 0.45,
                "max_velocity": 0.75,
                "minimum_height": 0.76,
            },
        }

        replace_reward_params = {"track_lin_vel_xy_exp", "world_forward_velocity_clip"}

        for reward_name, func in reward_functions.items():
            getattr(self.rewards, reward_name).func = func

        for reward_name, params in reward_params.items():
            reward_term = getattr(self.rewards, reward_name)
            if reward_name in replace_reward_params or reward_term.params is None:
                reward_term.params = dict(params)
            else:
                reward_term.params.update(params)

        for reward_name, weight in reward_weights.items():
            getattr(self.rewards, reward_name).weight = weight


@configclass
class KBotForwardFlatV32May31Top4Stage2EnvCfg(KBotForwardFlatV32May31Top4EnvCfg):
    """V3.2 stage-2 compatibility task: May 31 200-300 overtake and clearance pressure."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.foot_sagittal_separation_l1.weight = -4.0
        self.rewards.swing_foot_overtake_l1.weight = -100.0
        self.rewards.swing_foot_overtake_l1.params["target_length"] = 0.30
        self.rewards.swing_sole_clearance.weight = 1.0
