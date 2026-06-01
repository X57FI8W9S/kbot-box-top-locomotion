from __future__ import annotations

import math
import os
from isaaclab.managers import EventTermCfg as EventTerm
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
    REPO_ROOT,
)
from . import mdp


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
class KBotForwardFlatV1ReproEnvCfg(KBotForwardFlatEnvCfg):
    """V1 task with an optional USD override for artifact reproduction."""

    def __post_init__(self) -> None:
        super().__post_init__()
        usd_path = os.environ.get("KBOT_V1_REPRO_USD_PATH")
        if usd_path:
            self.scene.robot.spawn.usd_path = usd_path


@configclass
class KBotForwardFlatV1HybridWidthLaneEnvCfg(KBotForwardFlatEnvCfg):
    """V1 walking world with V2.5 foot-width and foot-lane pressures.

    This is a compatibility branch for V1 checkpoints. It intentionally keeps
    the V1 reset pose, command range, action scale, episode length, and gait
    rewards, while replacing only the lateral foot geometry targets.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        hip_axis_left_y = 0.15835
        hip_axis_right_y = -0.15805
        sole_center_offsets = [
            (0.03, -0.036528655, -0.0194786795),
            (0.03, -0.036528755, -0.0234786545),
        ]

        self.rewards.foot_lateral_spacing_l1.weight = -9.0
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

        self.rewards.foot_sole_lateral_lane_max_l1 = RewTerm(
            func=mdp.foot_sole_lateral_lane_max_l1,
            weight=-44.0,
            params={
                "target_left_y": hip_axis_left_y,
                "target_right_y": hip_axis_right_y,
                "tolerance": 0.008,
                "foot_local_offsets": sole_center_offsets,
                "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
            },
        )


@configclass
class KBotForwardFlatV1HybridWidthLaneEnvCfg_PLAY(KBotForwardFlatV1HybridWidthLaneEnvCfg):
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
class KBotForwardFlatV2ScratchHardEnvCfg(KBotForwardFlatV2EnvCfg):
    """Scratch probe: keep posture rewards but restore hard fall cutoffs."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.05, 0.15)
        self.rewards.forward_velocity_below_l2.weight = -4.0
        self.rewards.forward_velocity_below_l2.params["minimum_velocity"] = 0.05
        self.terminations.low_body = DoneTerm(func=mdp.root_height_below, params={"minimum_height": 0.55})
        self.terminations.bad_orientation = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": 0.95})


@configclass
class KBotForwardFlatV2ScratchStandEnvCfg(KBotForwardFlatV2EnvCfg):
    """Scratch probe: first learn tall quiet support before asking for walking."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.05)
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.alternating_foot_phase.weight = 0.0
        self.rewards.forward_velocity_below_l2.weight = 0.0
        self.rewards.foot_sagittal_separation_l1.weight = 0.0
        self.rewards.swing_foot_overtake_l1.weight = 0.0
        self.rewards.stance_foot_flat_l2.weight = -1.0
        self.rewards.base_height_l2.weight = -45.0
        self.rewards.low_body_l2.weight = -160.0
        self.rewards.upright_alive.weight = 10.0


@configclass
class KBotForwardFlatV2ScratchStandConservativeEnvCfg(KBotForwardFlatV2ScratchStandEnvCfg):
    """Scratch probe: standing-first plus smaller action scale and reset noise."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.scale = 0.10
        self.events.reset_base.params["pose_range"] = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.02, 0.02)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (-0.01, 0.01),
            "roll": (-0.01, 0.01),
            "pitch": (-0.01, 0.01),
            "yaw": (-0.01, 0.01),
        }
        self.events.reset_robot_joints.params["position_range"] = (0.99, 1.01)


@configclass
class KBotForwardFlatV2ScratchBalanceEnvCfg(KBotForwardFlatV2ScratchStandConservativeEnvCfg):
    """Scratch probe: balance-only survival before any gait shaping."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.actions.joint_pos.scale = 0.08
        self.episode_length_s = 4.0

        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.alternating_foot_phase.weight = 0.0
        self.rewards.foot_sagittal_separation_l1.weight = 0.0
        self.rewards.swing_foot_overtake_l1.weight = 0.0
        self.rewards.foot_lateral_lane_l1.weight = 0.0
        self.rewards.foot_lateral_spacing_l1.weight = -1.0
        self.rewards.foot_signed_lateral_clearance_l1.weight = -5.0
        self.rewards.leg_frontal_plane_l1.weight = 0.0
        self.rewards.foot_parallel_l2.weight = 0.0
        self.rewards.foot_toe_in_l2.weight = 0.0
        self.rewards.stance_foot_flat_l2.weight = 0.0

        self.rewards.base_height_l2.weight = -60.0
        self.rewards.base_height_l2.params["target_height"] = 0.78
        self.rewards.low_body_l2.weight = -240.0
        self.rewards.low_body_l2.params["minimum_height"] = 0.65
        self.rewards.upright_alive.weight = 20.0
        self.rewards.upright_alive.params["minimum_height"] = 0.65
        self.rewards.upright_alive.params["max_tilt"] = 0.25
        self.rewards.flat_orientation_l2.weight = -45.0
        self.rewards.root_lateral_tilt_l2.weight = -120.0
        self.rewards.root_lateral_tilt_ema_l2.weight = -500.0
        self.rewards.hip_roll_yaw_position_l2.weight = -6.0
        self.rewards.hip_roll_yaw_position_ema_l2.weight = -12.0
        self.rewards.hip_roll_position_ema_5cycle_l2.weight = 0.0
        self.rewards.termination_penalty.weight = -100.0

        self.terminations.low_body = DoneTerm(func=mdp.root_height_below, params={"minimum_height": 0.55})
        self.terminations.bad_orientation = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": 0.65})


@configclass
class KBotForwardFlatV2ScratchV1BootstrapEnvCfg(KBotForwardFlatEnvCfg):
    """Scratch bootstrap that mirrors the first V1 run that learned past falling."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1024
        self.episode_length_s = 3.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.10, 0.25)

        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.12)
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.10)
        self.rewards.action_rate_l2.weight = -0.025
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.40
        self.rewards.alternating_foot_phase.weight = 0.0
        self.rewards.alive.weight = 5.0
        self.rewards.base_height_l2.weight = -15.0
        self.rewards.base_height_l2.params["target_height"] = 0.78

        self.rewards.lateral_velocity_l2.weight = -2.0
        self.rewards.yaw_rate_l2.weight = -0.5
        self.rewards.root_lateral_tilt_l2.weight = 0.0
        self.rewards.root_lateral_tilt_ema_l2.weight = 0.0
        self.rewards.world_heading_l2.weight = 0.0
        self.rewards.forward_velocity_below_l2.weight = 0.0

        self.rewards.foot_lateral_spacing_l1.weight = 0.0
        self.rewards.foot_signed_lateral_clearance_l1.weight = 0.0
        self.rewards.foot_lateral_lane_l1.weight = 0.0
        self.rewards.foot_lateral_lane_max_l1.weight = 0.0
        self.rewards.leg_frontal_plane_l1.weight = 0.0
        self.rewards.left_leg_frontal_plane_l1.weight = 0.0
        self.rewards.right_leg_frontal_plane_l1.weight = 0.0
        self.rewards.max_leg_frontal_plane_l1.weight = 0.0
        self.rewards.foot_sagittal_separation_l1.weight = 0.0
        self.rewards.swing_foot_overtake_l1.weight = 0.0
        self.rewards.foot_parallel_l2.weight = 0.0
        self.rewards.foot_world_parallel_l2.weight = 0.0
        self.rewards.foot_world_parallel_max_l2.weight = 0.0
        self.rewards.foot_toe_in_l2.weight = 0.0
        self.rewards.foot_flat_l2.weight = 0.0
        self.rewards.stance_foot_flat_l2.weight = 0.0
        self.rewards.wobble_joint_vel_l2.weight = 0.0
        self.rewards.hip_roll_yaw_position_l2.weight = 0.0
        self.rewards.hip_roll_yaw_position_ema_l2.weight = 0.0

        self.rewards.low_body_l2.weight = -30.0
        self.rewards.low_body_l2.params["minimum_height"] = 0.45
        self.rewards.knee_extension_l1.weight = -80.0
        self.rewards.knee_extension_l1.params["min_bend"] = 0.50
        self.rewards.termination_penalty.weight = -500.0

        self.terminations.low_body = None
        self.terminations.bad_orientation = None


@configclass
class KBotForwardFlatV23ScratchV1BootstrapEnvCfg(KBotForwardFlatV2ScratchV1BootstrapEnvCfg):
    """V2.3 anti-fall bootstrap with the corrected hip-axis support width."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.foot_lateral_spacing_l1.weight = -0.25
        self.rewards.foot_lateral_spacing_l1.params["target_width"] = 0.3164
        self.rewards.foot_signed_lateral_clearance_l1.weight = -0.75
        self.rewards.foot_signed_lateral_clearance_l1.params["minimum_width"] = 0.24
        self.rewards.foot_lateral_lane_l1.weight = -0.20
        self.rewards.foot_lateral_lane_l1.params["target_left_y"] = 0.1582
        self.rewards.foot_lateral_lane_l1.params["target_right_y"] = -0.1582
        self.rewards.foot_lateral_lane_l1.params["tolerance"] = 0.08
        self.rewards.foot_lateral_lane_max_l1.weight = -0.05
        self.rewards.foot_lateral_lane_max_l1.params["target_left_y"] = 0.1582
        self.rewards.foot_lateral_lane_max_l1.params["target_right_y"] = -0.1582
        self.rewards.foot_lateral_lane_max_l1.params["tolerance"] = 0.06


@configclass
class KBotForwardFlatV2GaitTransitionEnvCfg(KBotForwardFlatV2ScratchV1BootstrapEnvCfg):
    """Continuation stage: turn the anti-fall seed into early forward gait gently."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 4.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.30)

        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.08)
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.action_rate_l2.weight = -0.04
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.35
        self.rewards.alternating_foot_phase.weight = 0.20

        self.rewards.lateral_velocity_l2.weight = -3.0
        self.rewards.yaw_rate_l2.weight = -1.5
        self.rewards.root_lateral_tilt_l2.weight = -10.0
        self.rewards.forward_velocity_below_l2.weight = -4.0
        self.rewards.forward_velocity_below_l2.params["minimum_velocity"] = 0.10

        self.rewards.foot_lateral_spacing_l1.weight = -1.0
        self.rewards.foot_signed_lateral_clearance_l1.weight = -2.0
        self.rewards.foot_signed_lateral_clearance_l1.params["minimum_width"] = 0.14
        self.rewards.foot_sagittal_separation_l1.weight = -1.5
        self.rewards.foot_sagittal_separation_l1.params["target_length"] = 0.14
        self.rewards.swing_foot_overtake_l1.weight = -2.0
        self.rewards.swing_foot_overtake_l1.params["target_length"] = 0.10
        self.rewards.foot_parallel_l2.weight = -0.3
        self.rewards.foot_flat_l2.weight = -0.2
        self.rewards.stance_foot_flat_l2.weight = -0.5

        self.rewards.base_height_l2.weight = -18.0
        self.rewards.base_height_l2.params["target_height"] = 0.78
        self.rewards.low_body_l2.weight = -40.0
        self.rewards.low_body_l2.params["minimum_height"] = 0.50
        self.rewards.knee_extension_l1.weight = -40.0
        self.rewards.knee_extension_l1.params["min_bend"] = 0.45


@configclass
class KBotForwardFlatV23GaitTransitionEnvCfg(KBotForwardFlatV2GaitTransitionEnvCfg):
    """V2.3 gait transition with hip-axis support-width targets."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.foot_lateral_spacing_l1.params["target_width"] = 0.3164
        self.rewards.foot_signed_lateral_clearance_l1.params["minimum_width"] = 0.24
        self.rewards.foot_lateral_lane_l1.weight = -0.3
        self.rewards.foot_lateral_lane_l1.params["target_left_y"] = 0.1582
        self.rewards.foot_lateral_lane_l1.params["target_right_y"] = -0.1582
        self.rewards.foot_lateral_lane_l1.params["tolerance"] = 0.08
        self.rewards.foot_lateral_lane_max_l1.weight = -0.1
        self.rewards.foot_lateral_lane_max_l1.params["target_left_y"] = 0.1582
        self.rewards.foot_lateral_lane_max_l1.params["target_right_y"] = -0.1582
        self.rewards.foot_lateral_lane_max_l1.params["tolerance"] = 0.06


@configclass
class KBotForwardFlatV2YawLateralTransitionEnvCfg(KBotForwardFlatV2GaitTransitionEnvCfg):
    """Continuation stage: keep early gait while reducing yaw, lateral drift, and roll bias."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.track_lin_vel_xy_exp.weight = 2.6
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.action_rate_l2.weight = -0.05

        self.rewards.lateral_velocity_l2.weight = -6.0
        self.rewards.yaw_rate_l2.weight = -4.0
        self.rewards.root_lateral_tilt_l2.weight = -25.0
        self.rewards.root_lateral_tilt_ema_l2.weight = -90.0
        self.rewards.root_lateral_tilt_ema_l2.params["tau_s"] = 2.0
        self.rewards.world_heading_l2.weight = -8.0

        self.rewards.foot_lateral_spacing_l1.weight = -2.0
        self.rewards.foot_signed_lateral_clearance_l1.weight = -4.0
        self.rewards.foot_signed_lateral_clearance_l1.params["minimum_width"] = 0.18
        self.rewards.foot_lateral_lane_l1.weight = -2.0
        self.rewards.foot_lateral_lane_l1.params["tolerance"] = 0.06
        self.rewards.foot_sagittal_separation_l1.weight = -2.0
        self.rewards.swing_foot_overtake_l1.weight = -3.0
        self.rewards.foot_parallel_l2.weight = -0.5
        self.rewards.foot_flat_l2.weight = -0.3
        self.rewards.stance_foot_flat_l2.weight = -0.8

        self.rewards.hip_roll_yaw_position_l2.weight = -2.0
        self.rewards.hip_roll_yaw_position_ema_l2.weight = -5.0
        self.rewards.hip_roll_yaw_position_ema_l2.params["tau_s"] = 2.0


@configclass
class KBotForwardFlatV2LateralCleanupEnvCfg(KBotForwardFlatV2YawLateralTransitionEnvCfg):
    """Continuation stage: reduce lateral path drift and edge/toe contacts."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.track_lin_vel_xy_exp.weight = 2.4
        self.rewards.track_ang_vel_z_exp.weight = 2.2
        self.rewards.action_rate_l2.weight = -0.055
        self.rewards.feet_air_time.weight = 1.2

        self.rewards.lateral_velocity_l2.weight = -7.0
        self.rewards.root_lateral_position_l2 = RewTerm(func=mdp.root_lateral_position_l2, weight=-8.0)
        self.rewards.yaw_rate_l2.weight = -4.5
        self.rewards.world_heading_l2.weight = -10.0
        self.rewards.root_lateral_tilt_l2.weight = -30.0
        self.rewards.root_lateral_tilt_ema_l2.weight = -110.0

        self.rewards.foot_lateral_spacing_l1.weight = -2.5
        self.rewards.foot_signed_lateral_clearance_l1.weight = -5.0
        self.rewards.foot_signed_lateral_clearance_l1.params["minimum_width"] = 0.20
        self.rewards.foot_lateral_lane_l1.weight = -3.0
        self.rewards.foot_lateral_lane_l1.params["tolerance"] = 0.045
        self.rewards.foot_lateral_lane_max_l1.weight = -1.0
        self.rewards.foot_lateral_lane_max_l1.params["tolerance"] = 0.035
        self.rewards.foot_parallel_l2.weight = -0.7
        self.rewards.foot_flat_l2.weight = -0.5
        self.rewards.stance_foot_flat_l2.weight = -1.2

        self.rewards.hip_roll_yaw_position_l2.weight = -2.5
        self.rewards.hip_roll_yaw_position_ema_l2.weight = -6.0


@configclass
class KBotForwardFlatV2HipContactCleanupEnvCfg(KBotForwardFlatV2LateralCleanupEnvCfg):
    """Continuation stage: reduce hip-roll bias while improving contact duty."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.track_lin_vel_xy_exp.weight = 2.3
        self.rewards.track_ang_vel_z_exp.weight = 2.2
        self.rewards.feet_air_time.weight = 1.35
        self.rewards.feet_air_time.params["threshold"] = 0.32
        self.rewards.action_rate_l2.weight = -0.06

        self.rewards.root_lateral_position_l2.weight = -6.0
        self.rewards.lateral_velocity_l2.weight = -6.5
        self.rewards.root_lateral_tilt_l2.weight = -35.0
        self.rewards.root_lateral_tilt_ema_l2.weight = -140.0
        self.rewards.root_lateral_tilt_ema_l2.params["tau_s"] = 2.5

        self.rewards.hip_roll_yaw_position_l2.weight = -3.5
        self.rewards.hip_roll_yaw_position_ema_l2.weight = -10.0
        self.rewards.hip_roll_yaw_position_ema_l2.params["tau_s"] = 2.5
        self.rewards.hip_roll_position_ema_5cycle_l2 = RewTerm(
            func=mdp.joint_position_ema_l2,
            weight=-28.0,
            params={"tau_s": 5.0, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_roll.*"])},
        )

        self.rewards.foot_lateral_lane_l1.weight = -2.5
        self.rewards.foot_lateral_lane_max_l1.weight = -1.2
        self.rewards.foot_sagittal_separation_l1.weight = -1.5
        self.rewards.swing_foot_overtake_l1.weight = -2.0
        self.rewards.foot_flat_l2.weight = -0.7
        self.rewards.stance_foot_flat_l2.weight = -1.8
        self.rewards.single_stance_foot_flat_l2 = RewTerm(
            func=mdp.single_stance_foot_flat_l2,
            weight=-1.2,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
            },
        )


@configclass
class KBotForwardFlatV2StepLengthCleanupEnvCfg(KBotForwardFlatV2LateralCleanupEnvCfg):
    """Fine-tune stage: lengthen tiny rapid steps while preserving straight, upright gait."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 2.4
        self.rewards.action_rate_l2.weight = -0.07
        self.rewards.feet_air_time.weight = 1.1
        self.rewards.feet_air_time.params["threshold"] = 0.18

        self.rewards.root_lateral_position_l2.weight = -9.0
        self.rewards.lateral_velocity_l2.weight = -7.5
        self.rewards.yaw_rate_l2.weight = -5.0
        self.rewards.world_heading_l2.weight = -12.0
        self.rewards.root_lateral_tilt_l2.weight = -35.0
        self.rewards.root_lateral_tilt_ema_l2.weight = -125.0

        self.rewards.base_height_l2.weight = -24.0
        self.rewards.base_height_l2.params["target_height"] = 0.80
        self.rewards.low_body_l2.weight = -55.0
        self.rewards.low_body_l2.params["minimum_height"] = 0.55

        self.rewards.foot_sagittal_separation_l1.weight = -3.5
        self.rewards.foot_sagittal_separation_l1.params["target_length"] = 0.08
        self.rewards.swing_foot_overtake_l1.weight = -5.0
        self.rewards.swing_foot_overtake_l1.params["target_length"] = 0.06
        self.rewards.swing_foot_overtake_l1.params["target_air_time"] = 0.20
        self.rewards.swing_foot_overtake_l1.params["grace_time"] = 0.04

        self.rewards.foot_lateral_spacing_l1.weight = -3.0
        self.rewards.foot_signed_lateral_clearance_l1.weight = -5.5
        self.rewards.foot_lateral_lane_l1.weight = -3.2
        self.rewards.foot_lateral_lane_max_l1.weight = -1.2
        self.rewards.foot_parallel_l2.weight = -0.8
        self.rewards.foot_flat_l2.weight = -0.6
        self.rewards.stance_foot_flat_l2.weight = -1.4

        self.rewards.hip_roll_yaw_position_l2.weight = -3.0
        self.rewards.hip_roll_yaw_position_ema_l2.weight = -8.0


@configclass
class KBotForwardFlatV2CadenceLengthCleanupEnvCfg(KBotForwardFlatV2LateralCleanupEnvCfg):
    """Fine-tune stage: slow the rapid shuffle while keeping the last good straightness."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.observations.policy.gait_phase.params["period_s"] = 1.20
        self.rewards.alternating_foot_phase.params["period_s"] = 1.20
        self.rewards.alternating_foot_phase.weight = 0.15

        self.rewards.track_lin_vel_xy_exp.weight = 2.8
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.08)
        self.rewards.track_ang_vel_z_exp.weight = 2.2
        self.rewards.action_rate_l2.weight = -0.055
        self.rewards.feet_air_time.weight = 1.35
        self.rewards.feet_air_time.params["threshold"] = 0.30

        self.rewards.root_lateral_position_l2.weight = -7.0
        self.rewards.lateral_velocity_l2.weight = -6.5
        self.rewards.yaw_rate_l2.weight = -4.2
        self.rewards.world_heading_l2.weight = -9.0
        self.rewards.root_lateral_tilt_l2.weight = -28.0
        self.rewards.root_lateral_tilt_ema_l2.weight = -105.0

        self.rewards.base_height_l2.weight = -20.0
        self.rewards.base_height_l2.params["target_height"] = 0.79
        self.rewards.low_body_l2.weight = -45.0
        self.rewards.low_body_l2.params["minimum_height"] = 0.54

        self.rewards.foot_sagittal_separation_l1.weight = -2.0
        self.rewards.foot_sagittal_separation_l1.params["target_length"] = 0.16
        self.rewards.swing_foot_overtake_l1.weight = -3.0
        self.rewards.swing_foot_overtake_l1.params["target_length"] = 0.12
        self.rewards.swing_foot_overtake_l1.params["target_air_time"] = 0.30
        self.rewards.swing_foot_overtake_l1.params["grace_time"] = 0.06

        self.rewards.foot_lateral_spacing_l1.weight = -2.5
        self.rewards.foot_lateral_spacing_l1.params["target_width"] = 0.3164
        self.rewards.foot_signed_lateral_clearance_l1.weight = -5.0
        self.rewards.foot_signed_lateral_clearance_l1.params["minimum_width"] = 0.24
        self.rewards.foot_lateral_lane_l1.weight = -2.8
        self.rewards.foot_lateral_lane_l1.params["target_left_y"] = 0.1582
        self.rewards.foot_lateral_lane_l1.params["target_right_y"] = -0.1582
        self.rewards.foot_lateral_lane_max_l1.weight = -1.0
        self.rewards.foot_lateral_lane_max_l1.params["target_left_y"] = 0.1582
        self.rewards.foot_lateral_lane_max_l1.params["target_right_y"] = -0.1582
        self.rewards.foot_parallel_l2.weight = -0.6
        self.rewards.foot_flat_l2.weight = -0.4
        self.rewards.stance_foot_flat_l2.weight = -1.0

        self.rewards.hip_roll_yaw_position_l2.weight = -2.5
        self.rewards.hip_roll_yaw_position_ema_l2.weight = -6.0


@configclass
class KBotForwardFlatV2HipAxisWidthCleanupEnvCfg(KBotForwardFlatV2LateralCleanupEnvCfg):
    """Fine-tune stage: move support width to hip axes without spending hip roll."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 2.4
        self.rewards.action_rate_l2.weight = -0.06
        self.rewards.feet_air_time.weight = 1.2

        self.rewards.root_lateral_position_l2.weight = -12.0
        self.rewards.lateral_velocity_l2.weight = -8.0
        self.rewards.yaw_rate_l2.weight = -5.5
        self.rewards.world_heading_l2.weight = -14.0
        self.rewards.root_lateral_tilt_l2.weight = -45.0
        self.rewards.root_lateral_tilt_ema_l2.weight = -180.0

        self.rewards.foot_lateral_spacing_l1.weight = -2.2
        self.rewards.foot_lateral_spacing_l1.params["target_width"] = 0.3164
        self.rewards.foot_signed_lateral_clearance_l1.weight = -4.5
        self.rewards.foot_signed_lateral_clearance_l1.params["minimum_width"] = 0.24
        self.rewards.foot_lateral_lane_l1.weight = -2.4
        self.rewards.foot_lateral_lane_l1.params["target_left_y"] = 0.1582
        self.rewards.foot_lateral_lane_l1.params["target_right_y"] = -0.1582
        self.rewards.foot_lateral_lane_max_l1.weight = -0.8
        self.rewards.foot_lateral_lane_max_l1.params["target_left_y"] = 0.1582
        self.rewards.foot_lateral_lane_max_l1.params["target_right_y"] = -0.1582

        self.rewards.foot_sagittal_separation_l1.weight = -1.2
        self.rewards.swing_foot_overtake_l1.weight = -1.6
        self.rewards.foot_parallel_l2.weight = -0.6
        self.rewards.foot_flat_l2.weight = -0.5
        self.rewards.stance_foot_flat_l2.weight = -1.2

        self.rewards.hip_roll_yaw_position_l2.weight = -5.5
        self.rewards.hip_roll_yaw_position_ema_l2.weight = -18.0
        self.rewards.hip_roll_yaw_position_ema_l2.params["tau_s"] = 2.5


@configclass
class KBotForwardFlatV2ScratchPoseBootstrapEnvCfg(KBotForwardFlatV2ScratchStandConservativeEnvCfg):
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
class KBotForwardFlatV24ScratchPoseBootstrapEnvCfg(KBotForwardFlatV2ScratchPoseBootstrapEnvCfg):
    """V2.4 scratch seed from the settled hand-authored standing pose."""

    pass


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
class KBotForwardFlatV25PoseGaitQualityEnvCfg(KBotForwardFlatV25ScratchPoseWidthBootstrapEnvCfg):
    """Continuation from V2.5 posed-start seed: improve swing quality without losing width."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.16)

        self.rewards.world_forward_velocity_clip.weight = 2.5
        self.rewards.world_forward_velocity_clip.params["max_velocity"] = 0.12
        self.rewards.action_rate_l2.weight = -0.30
        self.rewards.dof_acc_l2.weight = -5.0e-7
        self.rewards.wobble_joint_vel_l2.weight = -0.12
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
        self.rewards.swing_foot_overtake_l1.params["grace_time"] = 0.08
        self.rewards.valid_step_root_advance = RewTerm(
            func=mdp.valid_step_root_advance_reward,
            weight=4.0,
            params={
                "command_name": "base_velocity",
                "target_cycle_hz": 1.25,
                "min_step_advance": 0.020,
                "min_air_time": 0.12,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.step_advance_margin = RewTerm(
            func=mdp.step_advance_margin_reward,
            weight=float(os.environ.get("KBOT_STEP_MARGIN_WEIGHT", "8.0")),
            params={
                "command_name": "base_velocity",
                "target_cycle_hz": 1.25,
                "min_step_advance": 0.008,
                "max_step_advance": 0.12,
                "short_step_penalty_scale": 1.5,
                "min_step_duration": 0.07,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.dense_single_support_step_progress = RewTerm(
            func=mdp.dense_single_support_step_progress_reward,
            weight=float(os.environ.get("KBOT_DENSE_STEP_PROGRESS_WEIGHT", "6.0")),
            params={
                "command_name": "base_velocity",
                "target_cycle_hz": 1.25,
                "max_step_advance": 0.12,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        # Future swing-clearance margin target: 0.005 m. No z reward is active in this stage.
        self.rewards.supported_forward_velocity = RewTerm(
            func=mdp.supported_forward_velocity_reward,
            weight=1.5,
            params={
                "command_name": "base_velocity",
                "min_single_support_fraction": 0.20,
                "max_velocity": 0.12,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.walking_cycle_cadence_above_l2 = RewTerm(
            func=mdp.walking_cycle_cadence_above_l2,
            weight=-1.0,
            params={
                "max_cycle_hz": 1.25,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
            },
        )
        self.rewards.contact_chatter_l1 = RewTerm(
            func=mdp.contact_chatter_l1,
            weight=-6.0,
            params={
                "min_air_time": 0.12,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
            },
        )

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
        self.rewards.action_rate_l2.weight = -0.30
        self.rewards.stand_joint_position_l2.weight = -0.5


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
class KBotForwardFlatV25S42ChatterSuppressionEnvCfg(KBotForwardFlatV25PoseGaitQualityEnvCfg):
    """S4.2: suppress contact chatter from model_648 without training speed."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.contact_chatter_l1.weight = -10.0
        self.rewards.contact_chatter_l1.params["min_air_time"] = 0.10


@configclass
class KBotForwardFlatV25S42ChatterFrom648CompatEnvCfg(KBotForwardFlatV25PoseGaitQuality648CompatEnvCfg):
    """S4.2 branch from the frozen model_648 stack with only chatter added."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.contact_chatter_l1 = RewTerm(
            func=mdp.contact_chatter_l1,
            weight=-10.0,
            params={
                "min_air_time": 0.10,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
            },
        )


@configclass
class KBotForwardFlatV3HandTuned648EnvCfg(KBotForwardFlatV25PoseGaitQuality648CompatEnvCfg):
    """V3 scratch restart using the frozen model_648 reward topology.

    V3 intentionally starts from the code path that trained the V2.5
    `model_648.pt` seed, but as a fresh iteration-0 policy. It does not inherit
    the later S4 step-gate/cadence/chatter reward terms. The explicit weight
    assignments below intentionally match the 648-compatible baseline so this
    task behaves the same until those numbers are hand-tuned.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # V3 tuning block. These values match the saved model_648 reward
        # weights; edit only this block when making hand-tuned V3 branches.
        self.commands.base_velocity.ranges.lin_vel_x = (0.75, 0.75)
        self.rewards.track_lin_vel_xy_exp.func = mdp.upright_gated_track_lin_vel_xy_exp
        self.rewards.track_lin_vel_xy_exp.params["minimum_height"] = 0.76
        self.rewards.track_lin_vel_xy_exp.params["max_tilt"] = 0.45
        self.rewards.track_lin_vel_xy_exp.weight = 15.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.25
        self.rewards.dof_torques_l2.weight = -5.0e-5
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.action_rate_l2.weight = -0.09
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.flat_orientation_l2.weight = -20.0
        self.rewards.dof_pos_limits.weight = -2.0
        self.rewards.alive.weight = 1.0
        self.rewards.base_height_l2.weight = -35.0
        self.rewards.alternating_foot_phase.weight = 0.18
        self.rewards.lateral_velocity_l2.weight = -20.0
        self.rewards.yaw_rate_l2.weight = -20.0
        self.rewards.root_lateral_tilt_l2.weight = -90.0
        self.rewards.root_lateral_tilt_ema_l2.weight = -12000.0
        self.rewards.signed_joint_pair_ema_symmetry_l2 = RewTerm(
            func=mdp.signed_joint_pair_ema_symmetry_l2,
            weight=-90.0,
            params={
                "tau_s": 1.0,
                "joint_pairs": [
                    ("left_hip_pitch_04", "right_hip_pitch_04", -1.0),
                    ("left_hip_roll_03", "right_hip_roll_03", -1.0),
                    ("left_hip_yaw_03", "right_hip_yaw_03", -1.0),
                    ("left_knee_04", "right_knee_04", -1.0),
                    ("left_ankle_02", "right_ankle_02", -1.0),
                ],
            },
        )
        self.rewards.world_heading_l2.weight = -90.0
        self.rewards.backward_velocity_l2.weight = -2.0
        self.rewards.forward_velocity_below_l2.weight = -6.0
        self.rewards.foot_lateral_spacing_l1.weight = -30.0
        self.rewards.foot_signed_lateral_clearance_l1.weight = -24.0
        self.rewards.foot_lateral_lane_l1.weight = -4.0
        self.rewards.foot_lateral_lane_max_l1.weight = -2.0
        self.rewards.leg_frontal_plane_l1.weight = 0.0
        self.rewards.left_leg_frontal_plane_l1.weight = -4.0
        self.rewards.right_leg_frontal_plane_l1.weight = -4.0
        self.rewards.max_leg_frontal_plane_l1.weight = -10.0
        self.rewards.foot_sagittal_separation_l1.weight = -12.0
        self.rewards.foot_sagittal_separation_l1.params["target_length"] = 0.60
        self.rewards.swing_foot_overtake_l1.weight = -150.0
        self.rewards.swing_foot_overtake_l1.params["target_length"] = 0.30
        self.rewards.foot_parallel_l2.weight = -30.0
        self.rewards.foot_world_parallel_l2.weight = 0.0
        self.rewards.foot_world_parallel_max_l2.weight = -0.8
        self.rewards.foot_toe_in_l2.weight = 0.0
        self.rewards.foot_flat_l2.weight = 0.0
        self.rewards.stance_foot_flat_l2.weight = -1.2
        self.rewards.wobble_joint_vel_l2.weight = -0.04
        self.rewards.hip_roll_yaw_position_l2.weight = 0.0
        self.rewards.hip_roll_yaw_position_ema_l2.weight = 0.0
        self.rewards.low_body_l2.weight = -120.0
        self.rewards.knee_extension_l1.weight = 0.0
        self.rewards.termination_penalty.weight = -750.0
        self.rewards.upright_alive.weight = 8.0
        self.rewards.foot_sole_lateral_lane_max_l1.weight = -100.0
        self.rewards.leg_frontal_sole_plane_max_l1.weight = -14.0
        self.rewards.hip_roll_position_ema_5cycle_l2.weight = 0.0
        self.rewards.centered_joint_target_position_l2.weight = 0.0
        self.rewards.stand_joint_position_l2.weight = -1.0
        self.rewards.world_forward_velocity_below_l2.weight = -24.0
        self.rewards.forward_velocity_below_l2.params["minimum_velocity"] = 0.07
        self.rewards.world_forward_velocity_below_l2.params["minimum_velocity"] = 0.05
        self.rewards.world_forward_velocity_clip.func = mdp.upright_gated_world_forward_velocity_clip
        self.rewards.world_forward_velocity_clip.params["minimum_height"] = 0.76
        self.rewards.world_forward_velocity_clip.params["max_tilt"] = 0.45
        self.rewards.world_forward_velocity_clip.weight = 15
        self.rewards.world_forward_velocity_clip.params["max_velocity"] = 0.75
        self.rewards.root_lateral_position_l2.weight = -200.0
        self.rewards.lateral_away_from_center_l2 = RewTerm(func=mdp.lateral_away_from_center_l2, weight=-600.0)
        self.rewards.contact_duty_symmetry_l2 = RewTerm(
            func=mdp.contact_duty_symmetry_l2,
            weight=-10.0,
            params={
                "tau_s": 1.0,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
            },
        )
        self.rewards.alternating_step_symmetry_l2 = RewTerm(
            func=mdp.alternating_step_symmetry_l2,
            weight=-8.0,
            params={
                "tau_s": 1.0,
                "advance_scale": 0.08,
                "duration_scale": 0.20,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
            },
        )
        self.rewards.swing_sole_clearance = RewTerm(
            func=mdp.swing_sole_clearance_reward,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "target_height": 0.010,
                "drag_floor": 0.002,
                "drag_weight": 3.0,
                "minimum_height": 0.76,
                "max_tilt": 0.45,
                "foot_local_offsets": [
                    (0.03, -0.036528655, -0.0194786795),
                    (0.03, -0.036528755, -0.0234786545),
                ],
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
                "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
            },
        )


@configclass
class KBotForwardFlatV2ScratchActionBootstrapEnvCfg(KBotForwardFlatV2ScratchV1BootstrapEnvCfg):
    """Scratch bootstrap with a short V1-derived recovery-action prior."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.early_action_sequence_l2 = RewTerm(
            func=mdp.early_action_sequence_l2,
            weight=-1.0,
            params={
                "duration_s": 0.40,
                "targets": [
                    [0.2099, -0.1198, 0.1361, 1.2101, 0.4279, -0.6150, -2.7360, 3.1930, -2.3437, -0.8811],
                    [5.1104, -2.9871, 1.3583, -0.7267, -0.1990, -1.2123, 5.0458, -1.9918, -5.4884, 6.2425],
                    [1.1912, -2.8021, -5.1936, -0.8531, -4.0128, -0.1630, -1.4994, -3.4933, -3.1833, 4.3908],
                ],
            },
        )


@configclass
class KBotForwardFlatV2ScratchActionBootstrapStrongEnvCfg(KBotForwardFlatV2ScratchActionBootstrapEnvCfg):
    """Scratch bootstrap that makes the first-second recovery action prior dominant."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.early_action_sequence_l2.weight = -20.0
        self.rewards.early_action_sequence_l2.params["duration_s"] = 1.0
