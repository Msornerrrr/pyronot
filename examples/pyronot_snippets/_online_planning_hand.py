from typing import Sequence, Optional

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyronot as pk


def solve_online_planning(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    target_position: onp.ndarray,
    target_wxyz: onp.ndarray,
    timesteps: int,
    dt: float,
    start_cfg: onp.ndarray,
    prev_sols: onp.ndarray,
    hand_target: Optional[onp.ndarray] = None,
) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """Solve online planning with collision."""

    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_indices = [robot.links.names.index(target_link_name)]

    target_poses = jaxlie.SE3(
        jnp.concatenate([jnp.array(target_wxyz), jnp.array(target_position)], axis=-1)
    )
    target_links = jnp.array(target_link_indices)

    # Warm start: use previous solution shifted by one step.
    timesteps = timesteps + 1  # for start pose cost.

    sol_traj, sol_pos, sol_wxyz = _solve_online_planning_jax(
        robot,
        robot_coll,
        world_coll,
        target_poses,
        target_links,
        timesteps,
        dt,
        jnp.array(start_cfg),
        jnp.concatenate([prev_sols, prev_sols[-1:]], axis=0),
        hand_target,
    )
    sol_traj = sol_traj[1:]
    sol_pos = sol_pos[1:]
    sol_wxyz = sol_wxyz[1:]

    return onp.array(sol_traj), onp.array(sol_pos), onp.array(sol_wxyz)


@jdc.jit
def _solve_online_planning_jax(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_poses: jaxlie.SE3,
    target_links: jnp.ndarray,
    timesteps: jdc.Static[int],
    dt: float,
    start_cfg: jnp.ndarray,
    prev_sols: jnp.ndarray,
    hand_target: Optional[onp.ndarray] = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    ARM_DOF = 7
    HAND_DOF = 16
    # arm_joint_indices = jnp.arange(0, ARM_DOF)
    hand_joint_indices = jnp.arange(ARM_DOF, ARM_DOF + HAND_DOF)

    num_targets = len(target_links)

    def batched_rplus(
        pose: jaxlie.SE3,
        delta: jax.Array,
    ) -> jaxlie.SE3:
        return jax.vmap(jaxlie.manifold.rplus)(pose, delta.reshape(num_targets, -1))

    # Custom SE3 variable to batch across multiple joint targets.
    # This is not to be confused with SE3Vars with ids, which we use here for timesteps.
    class BatchedSE3Var(  # pylint: disable=missing-class-docstring
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: jaxlie.SE3.identity((num_targets,)),
        retract_fn=batched_rplus,
        tangent_dim=jaxlie.SE3.tangent_dim * num_targets,
    ): ...

    # --- Define Variables ---
    traj_var = robot.joint_var_cls(jnp.arange(0, timesteps))
    traj_var_prev = robot.joint_var_cls(jnp.arange(0, timesteps - 1))
    traj_var_next = robot.joint_var_cls(jnp.arange(1, timesteps))
    pose_var = BatchedSE3Var(jnp.arange(0, timesteps))
    pose_var_prev = BatchedSE3Var(jnp.arange(0, timesteps - 1))
    pose_var_next = BatchedSE3Var(jnp.arange(1, timesteps))

    init_pose_vals = jaxlie.SE3(
        robot.forward_kinematics(prev_sols)[..., target_links, :]
    )

    # --- Define Costs ---
    factors: list[jaxls.Cost] = []  # Changed type hint to jaxls.Cost

    @jaxls.Cost.factory(name="SE3PoseMatchJointCost")
    def match_joint_to_pose_cost(
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[jnp.ndarray],
        pose_var: BatchedSE3Var,
    ):
        joint_cfg = vals[joint_var]
        target_pose = vals[pose_var]
        Ts_joint_world = robot.forward_kinematics(joint_cfg)
        residual = (
            (jaxlie.SE3(Ts_joint_world[..., target_links, :])).inverse() @ (target_pose)
        ).log()
        return residual.flatten() * 100.0

    @jaxls.Cost.factory(name="SE3SmoothnessCost")
    def pose_smoothness_cost(
        vals: jaxls.VarValues,
        pose_var: BatchedSE3Var,
        pose_var_prev: BatchedSE3Var,
    ):
        return (vals[pose_var].inverse() @ vals[pose_var_prev]).log().flatten() * 1.0

    @jaxls.Cost.factory(name="SE3PoseMatchCost")
    def pose_match_cost(
        vals: jaxls.VarValues,
        pose_var: BatchedSE3Var,
    ):
        return (
            (vals[pose_var].inverse() @ target_poses).log()
            * jnp.array([50.0] * 3 + [20.0] * 3)
        ).flatten()

    @jaxls.Cost.factory(name="MatchStartPoseCost")
    def match_start_pose_cost(
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[jnp.ndarray],
    ):
        return (vals[joint_var] - start_cfg).flatten() * 100.0

    # Add pose costs.
    factors.extend(
        [
            pose_match_cost(
                BatchedSE3Var(timesteps - 1),
            ),
            pose_smoothness_cost(
                pose_var_next,
                pose_var_prev,
            ),
        ]
    )

    # Need to constrain the start joint cfg.
    factors.append(match_start_pose_cost(robot.joint_var_cls(0)))

    # Add joint costs.
    # Technically we should add this as a constraint, but the augmented lagrangian formulation
    # requires us to perform multiple outer loop iterations to converge, which is slow. :(
    factors.extend(
        [
            match_joint_to_pose_cost(
                traj_var,
                pose_var,
            ),
            pk.costs.smoothness_cost(
                traj_var_prev,
                traj_var_next,
                weight=10.0,
            ),
            pk.costs.limit_velocity_cost(
                jax.tree.map(lambda x: x[None], robot),
                traj_var_prev,
                traj_var_next,
                weight=1.0,
                dt=dt,
            ),
            pk.costs.limit_cost(
                jax.tree.map(lambda x: x[None], robot),
                traj_var,
                weight=100.0,
            ),
            pk.costs.rest_cost(
                traj_var,
                jnp.array(traj_var.default_factory())[None],
                weight=0.5,
            ),
            pk.costs.manipulability_cost(
                jax.tree.map(lambda x: x[None], robot),
                traj_var,
                weight=0.01,
                target_link_indices=target_links,
            ),
            pk.costs.self_collision_cost(
                jax.tree.map(lambda x: x[None], robot),
                jax.tree.map(lambda x: x[None], robot_coll),
                traj_var,
                weight=10.0,
                margin=0.02,
            ),
        ]
    )
    factors.extend(
        [
            pk.costs.world_collision_cost(
                jax.tree.map(lambda x: x[None], robot),
                jax.tree.map(lambda x: x[None], robot_coll),
                traj_var,
                jax.tree.map(lambda x: x[None], obs),
                weight=20.0,
                margin=0.01,
            )
            for obs in world_coll
        ]
    )

    @jaxls.Cost.factory(name="HandTargetJointCost")
    def hand_target_cost(
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[jnp.ndarray],
    ):
        q = vals[joint_var]
        q_hand = q[hand_joint_indices]
        return (q_hand - hand_target).flatten() * 5.0

    if hand_target is not None:
        hand_target = jnp.asarray(hand_target)
        factors.append(
            hand_target_cost(traj_var)
        )

    solution = (
        jaxls.LeastSquaresProblem(factors, [traj_var, pose_var])
        .analyze()
        .solve(
            verbose=False,
            initial_vals=jaxls.VarValues.make(
                (traj_var.with_value(prev_sols), pose_var.with_value(init_pose_vals))
            ),
            termination=jaxls.TerminationConfig(max_iterations=20),
        )
    )
    pose_traj = solution[pose_var]
    return (
        solution[traj_var],
        pose_traj.translation(),
        pose_traj.rotation().wxyz,
    )
