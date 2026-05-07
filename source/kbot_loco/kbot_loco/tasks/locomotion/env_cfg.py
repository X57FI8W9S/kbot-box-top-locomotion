from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as base_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .assets import KBOT_CFG, REPO_ROOT
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
        self.rewards.mirrored_joint_position_l2 = RewTerm(
            func=mdp.mirrored_joint_position_l2,
            weight=-3.0,
            params={
                "joint_pairs": [
                    ("left_hip_pitch_04", "right_hip_pitch_04", 1.0),
                    ("left_hip_roll_03", "right_hip_roll_03", -1.0),
                    ("left_hip_yaw_03", "right_hip_yaw_03", -1.0),
                    ("left_knee_04", "right_knee_04", -1.0),
                    ("left_ankle_02", "right_ankle_02", -1.0),
                ],
            },
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
class KBotForwardFlatV2ScratchPoseBootstrapEnvCfg(KBotForwardFlatV2ScratchStandConservativeEnvCfg):
    """Scratch bootstrap from a hand-authored V1-derived standing pose."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1024
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.733)
        self.scene.robot.init_state.joint_pos.update(
            {
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
        self.rewards.base_height_l2.params["target_height"] = 0.733
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
