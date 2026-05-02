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

        self.commands.base_velocity.ranges.lin_vel_x = (0.30, 0.50)
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)

        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.feet_air_time.weight = 1.25
        self.rewards.alternating_foot_phase.weight = 0.25
        self.rewards.upright_alive = RewTerm(
            func=mdp.upright_alive,
            weight=2.0,
            params={"minimum_height": 0.55, "max_tilt": 0.45},
        )

        self.rewards.flat_orientation_l2.weight = -15.0
        self.rewards.lateral_velocity_l2.weight = -5.0
        self.rewards.yaw_rate_l2.weight = -5.0
        self.rewards.world_heading_l2.weight = -20.0
        self.rewards.root_lateral_tilt_l2.weight = -70.0
        self.rewards.root_lateral_tilt_ema_l2.weight = -350.0
        self.rewards.root_lateral_tilt_ema_l2.params["tau_s"] = 2.5

        self.rewards.forward_velocity_below_l2.weight = -16.0
        self.rewards.forward_velocity_below_l2.params["minimum_velocity"] = 0.25
        self.rewards.foot_lateral_spacing_l1.weight = -5.0
        self.rewards.foot_signed_lateral_clearance_l1.weight = -20.0
        self.rewards.foot_lateral_lane_l1.weight = -5.0
        self.rewards.foot_lateral_lane_l1.params["tolerance"] = 0.04
        self.rewards.foot_lateral_lane_max_l1.weight = 0.0

        self.rewards.leg_frontal_plane_l1.weight = -4.0
        self.rewards.leg_frontal_plane_l1.params["tolerance"] = 0.04
        self.rewards.left_leg_frontal_plane_l1.weight = 0.0
        self.rewards.right_leg_frontal_plane_l1.weight = 0.0
        self.rewards.max_leg_frontal_plane_l1.weight = 0.0

        self.rewards.foot_sagittal_separation_l1.weight = -3.0
        self.rewards.foot_sagittal_separation_l1.params["target_length"] = 0.22
        self.rewards.swing_foot_overtake_l1.weight = -10.0
        self.rewards.swing_foot_overtake_l1.params["target_length"] = 0.18
        self.rewards.foot_parallel_l2.weight = -1.0
        self.rewards.foot_toe_in_l2.weight = -6.0
        self.rewards.foot_flat_l2.weight = -0.35
        self.rewards.stance_foot_flat_l2.weight = -2.0

        self.rewards.hip_roll_yaw_position_l2.weight = -8.0
        self.rewards.hip_roll_yaw_position_ema_l2.weight = -24.0
        self.rewards.hip_roll_yaw_position_ema_l2.params["tau_s"] = 2.5
        self.rewards.hip_roll_position_ema_5cycle_l2 = RewTerm(
            func=mdp.joint_position_ema_l2,
            weight=-90.0,
            params={"tau_s": 5.0, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_roll.*"])},
        )

        self.rewards.low_body_l2.weight = -40.0
        self.rewards.low_body_l2.params["minimum_height"] = 0.50
        self.rewards.knee_extension_l1.weight = -18.0
        self.rewards.knee_extension_l1.params["min_bend"] = 0.35

        self.terminations.low_body = DoneTerm(func=mdp.root_height_below, params={"minimum_height": 0.42})
        self.terminations.bad_orientation = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": 0.95})


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
