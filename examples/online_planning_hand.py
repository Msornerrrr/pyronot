#!/usr/bin/env python3
"""
Online Planning + Viser hand joint goal sliders.
- Left: Viser transform controls for IK target (position + orientation)
- Obstacle: sphere transform controls
- NEW: hand joint goal sliders that feed into the optimizer as hand_target
"""

import time
import numpy as np
import pyronot as pk
import viser
import yourdfpy
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

from pyronot.collision import HalfSpace, RobotCollisionSpherized, Sphere
import pyronot_snippets as pks


ARM_DOF = 7  # xArm7


def _quat_wxyz_from_rpy(rpy_xyz: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_euler("xyz", rpy_xyz).as_quat()
    return np.roll(quat_xyzw, 1)  # xyzw -> wxyz


def main():
    urdf_path = "/home/slurmlab/hao/pyronot/third_party/foam/assets/xarm7_leaphand/xarm7_leap_right_spheres.urdf"
    urdf = yourdfpy.URDF.load(urdf_path)

    target_link_name = "link7"
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollisionSpherized.from_urdf(urdf)

    # World collision
    plane_coll = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    # front_wall_coll = HalfSpace.from_point_and_normal(
    #     np.array([0.7, 0.0, 0.0]),   # point on the plane: x = 0.7
    #     np.array([-1.0, 0.0, 0.0]),  # normal points toward -x
    # )
    sphere_coll = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.05])
    )

    # Planning params
    len_traj, dt = 5, 0.1

    # ----------------------------
    # Viser setup
    # ----------------------------
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # IK target controls
    init_wxyz = _quat_wxyz_from_rpy(np.array([3.14, 0.0, 0.0]))
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.3, 0.0, 0.5), wxyz=init_wxyz
    )

    # Obstacle controls
    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.2, position=(0.4, 0.3, 0.4)
    )
    server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=sphere_coll.to_trimesh())

    # Planned trajectory frames
    horizon_vis = 25
    target_frame_handle = server.scene.add_batched_axes(
        "target_frame",
        axes_length=0.05,
        axes_radius=0.005,
        batched_positions=np.zeros((horizon_vis, 3)),
        batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]] * horizon_vis),
    )

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # ----------------------------
    # Initialize trajectory warm start
    # ----------------------------
    sol_traj = np.array(robot.joint_var_cls.default_factory())[None].repeat(len_traj, axis=0)

    # Split dims
    total_dof = sol_traj.shape[-1]
    assert total_dof > ARM_DOF, f"Expected arm+hand DOF, got total_dof={total_dof}"
    hand_dof = total_dof - ARM_DOF

    # ----------------------------
    # NEW: Hand goal sliders (Viser GUI)
    # ----------------------------
    hand_folder = server.gui.add_folder("Hand goal (joint targets)")

    # Initialize hand target to current warm-start hand configuration
    hand_target = sol_traj[0, ARM_DOF:].copy()

    # Create one slider per hand joint
    # NOTE: slider ranges here are generic; adjust if you know Allegro limits.
    sliders = []
    for i in range(hand_dof):
        s = server.gui.add_slider(
            label=f"hand_q[{i}]",
            min=-1.5,
            max=1.5,
            step=0.01,
            initial_value=float(hand_target[i]),
        )
        sliders.append(s)

    # Convenience buttons
    def _set_sliders_from_current():
        nonlocal hand_target
        hand_target = sol_traj[0, ARM_DOF:].copy()
        for i, s in enumerate(sliders):
            s.value = float(hand_target[i])

    def _open_hand():
        for s in sliders:
            s.value = 0.0

    server.gui.add_button("Copy current -> target").on_click(lambda _: _set_sliders_from_current())
    server.gui.add_button("Set target = zeros").on_click(lambda _: _open_hand())

    # Optional: show current hand values read from solution
    show_current = server.gui.add_checkbox("Show current hand q (read-only)", initial_value=False)
    current_readouts = []
    for i in range(hand_dof):
        n = server.gui.add_number(f"cur_hand_q[{i}]", float(hand_target[i]), disabled=True)
        current_readouts.append(n)

    # ----------------------------
    # Main loop
    # ----------------------------
    while True:
        start_time = time.time()

        # Update obstacle geometry
        sphere_coll_world_current = sphere_coll.transform_from_wxyz_position(
            wxyz=np.array(sphere_handle.wxyz),
            position=np.array(sphere_handle.position),
        )
        world_coll_list = [plane_coll, sphere_coll_world_current]

        # Read target pose from Viser
        target_pos = np.array(ik_target_handle.position, dtype=float)
        target_wxyz = np.array(ik_target_handle.wxyz, dtype=float)

        # Read hand goal from sliders
        hand_target = np.array([s.value for s in sliders], dtype=float)

        # Solve (requires your updated solve_online_planning signature)
        sol_traj, sol_pos, sol_wxyz = pks.solve_online_planning_hand(
            robot=robot,
            robot_coll=robot_coll,
            world_coll=world_coll_list,
            target_link_name=target_link_name,
            target_position=target_pos,
            target_wxyz=target_wxyz,
            timesteps=len_traj,
            dt=dt,
            start_cfg=sol_traj[0],
            prev_sols=sol_traj,
            hand_target=hand_target,          # NEW
        )

        # Timing display
        elapsed_ms = (time.time() - start_time) * 1000.0
        timing_handle.value = 0.99 * float(timing_handle.value) + 0.01 * elapsed_ms

        # Update robot visualization with first step
        urdf_vis.update_cfg(sol_traj[0])

        # Update readouts
        if show_current.value:
            cur_hand = sol_traj[0, ARM_DOF:]
            for i, n in enumerate(current_readouts):
                n.value = float(cur_hand[i])

        # Trajectory frame visualization (pad/truncate)
        pos = np.array(sol_pos)
        wxyz = np.array(sol_wxyz)
        if pos.shape[0] < horizon_vis:
            pad = horizon_vis - pos.shape[0]
            pos = np.concatenate([pos, np.repeat(pos[-1:, :], pad, axis=0)], axis=0)
            wxyz = np.concatenate([wxyz, np.repeat(wxyz[-1:, :], pad, axis=0)], axis=0)
        else:
            pos = pos[:horizon_vis]
            wxyz = wxyz[:horizon_vis]

        if hasattr(target_frame_handle, "batched_positions"):
            target_frame_handle.batched_positions = pos  # type: ignore[attr-defined]
            target_frame_handle.batched_wxyzs = wxyz     # type: ignore[attr-defined]
        else:
            target_frame_handle.positions_batched = pos  # type: ignore[attr-defined]
            target_frame_handle.wxyzs_batched = wxyz     # type: ignore[attr-defined]


if __name__ == "__main__":
    main()