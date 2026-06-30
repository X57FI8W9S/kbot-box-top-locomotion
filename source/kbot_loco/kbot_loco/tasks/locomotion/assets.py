from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


REPO_ROOT = Path(__file__).resolve().parents[5]
KBOT_USD_PATH = REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top3.usd"
KBOT_PADS_USD_PATH = REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top3_pads.usd"
KBOT_TOP4_USD_PATH = REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top4.usd"
ISAACLAB_IMPLICIT_GAIN_SCALE = 57.3

HIP_PITCH_KNEE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*hip_pitch.*", ".*knee.*"],
    effort_limit=120.0,
    saturation_effort=120.0,
    velocity_limit=6.283,
    stiffness={".*": 45.0},
    damping={".*": 4.0},
)

HIP_ROLL_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*hip_roll.*"],
    effort_limit=60.0,
    saturation_effort=60.0,
    velocity_limit=6.283,
    stiffness={".*": 35.0},
    damping={".*": 3.0},
)

HIP_YAW_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*hip_yaw.*"],
    effort_limit=60.0,
    saturation_effort=60.0,
    velocity_limit=6.283,
    stiffness={".*": 25.0},
    damping={".*": 2.0},
)

ANKLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*ankle.*"],
    effort_limit=17.0,
    saturation_effort=17.0,
    velocity_limit=12.566,
    stiffness={".*": 12.0},
    damping={".*": 1.0},
)

IMPLICIT_HIP_PITCH_KNEE_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*hip_pitch.*", ".*knee.*"],
    effort_limit_sim=120.0,
    velocity_limit_sim=6.283,
    stiffness={".*": 45.0 * ISAACLAB_IMPLICIT_GAIN_SCALE},
    damping={".*": 4.0 * ISAACLAB_IMPLICIT_GAIN_SCALE},
)

IMPLICIT_HIP_ROLL_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*hip_roll.*"],
    effort_limit_sim=60.0,
    velocity_limit_sim=6.283,
    stiffness={".*": 35.0 * ISAACLAB_IMPLICIT_GAIN_SCALE},
    damping={".*": 3.0 * ISAACLAB_IMPLICIT_GAIN_SCALE},
)

IMPLICIT_HIP_YAW_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*hip_yaw.*"],
    effort_limit_sim=60.0,
    velocity_limit_sim=6.283,
    stiffness={".*": 25.0 * ISAACLAB_IMPLICIT_GAIN_SCALE},
    damping={".*": 2.0 * ISAACLAB_IMPLICIT_GAIN_SCALE},
)

IMPLICIT_ANKLE_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*ankle.*"],
    effort_limit_sim=17.0,
    velocity_limit_sim=12.566,
    stiffness={".*": 12.0 * ISAACLAB_IMPLICIT_GAIN_SCALE},
    damping={".*": 1.0 * ISAACLAB_IMPLICIT_GAIN_SCALE},
)


def _spawn_cfg(usd_path: Path) -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    )


KBOT_CFG = ArticulationCfg(
    spawn=_spawn_cfg(KBOT_USD_PATH),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            "left_hip_pitch_04": 0.0,
            "right_hip_pitch_04": 0.0,
            "left_hip_roll_03": 0.0,
            "right_hip_roll_03": 0.0,
            "left_hip_yaw_03": 0.0,
            "right_hip_yaw_03": 0.0,
            "left_knee_04": 0.75,
            "right_knee_04": -0.75,
            "left_ankle_02": 0.0,
            "right_ankle_02": 0.0,
        },
    ),
    actuators={
        "hip_pitch_knee": HIP_PITCH_KNEE_ACTUATOR_CFG,
        "hip_roll": HIP_ROLL_ACTUATOR_CFG,
        "hip_yaw": HIP_YAW_ACTUATOR_CFG,
        "ankles": ANKLE_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)

KBOT_TOP4_CFG = ArticulationCfg(
    spawn=_spawn_cfg(KBOT_TOP4_USD_PATH),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8565),
        joint_pos={
            "left_hip_pitch_04": 0.296467274,
            "left_hip_roll_03": 0.000730005,
            "left_hip_yaw_03": -0.000167742,
            "left_knee_04": 0.527403533,
            "left_ankle_02": -0.226248741,
            "right_hip_pitch_04": -0.296534926,
            "right_hip_roll_03": 0.000735207,
            "right_hip_yaw_03": -0.000100786,
            "right_knee_04": -0.528043807,
            "right_ankle_02": 0.226889685,
        },
    ),
    actuators={
        "hip_pitch_knee": HIP_PITCH_KNEE_ACTUATOR_CFG,
        "hip_roll": HIP_ROLL_ACTUATOR_CFG,
        "hip_yaw": HIP_YAW_ACTUATOR_CFG,
        "ankles": ANKLE_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)

KBOT_PADS_CFG = ArticulationCfg(
    spawn=_spawn_cfg(KBOT_PADS_USD_PATH),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.88),
        joint_pos={
            "left_hip_pitch_04": 0.0,
            "right_hip_pitch_04": 0.0,
            "left_hip_roll_03": 0.0,
            "right_hip_roll_03": 0.0,
            "left_hip_yaw_03": 0.0,
            "right_hip_yaw_03": 0.0,
            "left_knee_04": 0.75,
            "right_knee_04": -0.75,
            "left_ankle_02": 0.0,
            "right_ankle_02": 0.0,
        },
    ),
    actuators={
        "hip_pitch_knee": HIP_PITCH_KNEE_ACTUATOR_CFG,
        "hip_roll": HIP_ROLL_ACTUATOR_CFG,
        "hip_yaw": HIP_YAW_ACTUATOR_CFG,
        "ankles": ANKLE_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)
