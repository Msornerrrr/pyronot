#!/usr/bin/env python3
import time
from typing import Any, Dict, List, Optional

import zmq
import numpy as np
import pyronot as pk
import viser
import yourdfpy
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

from pyronot.collision import RobotCollisionSpherized, HalfSpace, Sphere
import pyronot_snippets as pks


ARM_DOF = 7  # xArm7


def _quat_wxyz_from_rpy(rpy_xyz: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_euler("xyz", rpy_xyz).as_quat()
    return np.roll(quat_xyzw, 1)


# ============================================================
# World collision (Pyroki)
# ============================================================

def _make_world_coll(
    req_world: Optional[List[Dict[str, Any]]],
    plane_default: HalfSpace,
    sphere_template: Sphere,
    viser_sphere_handle=None,
) -> List[pk.collision.CollGeom]:
    world_coll: List[pk.collision.CollGeom] = []

    if req_world is None:
        world_coll.append(plane_default)
        if viser_sphere_handle is not None:
            sphere_world = sphere_template.transform_from_wxyz_position(
                wxyz=np.array(viser_sphere_handle.wxyz),
                position=np.array(viser_sphere_handle.position),
            )
            world_coll.append(sphere_world)
        return world_coll

    for obj in req_world:
        typ = obj.get("type", "")
        if typ == "plane":
            normal = np.array(obj.get("normal", [0, 0, 1]), dtype=float)
            point = normal * float(obj.get("offset", 0.0))
            world_coll.append(HalfSpace.from_point_and_normal(point, normal))
        elif typ == "sphere":
            center = np.array(obj["center"], dtype=float)
            radius = float(obj["radius"])
            world_coll.append(
                Sphere.from_center_and_radius(center, np.array([radius]))
            )

    return world_coll


# ============================================================
# World visualization (Viser)
# ============================================================

def update_world_visualization(
    server: viser.ViserServer,
    req_world: List[Dict[str, Any]],
    sphere_template: Sphere,
    world_vis_handles: Dict[str, Any],
):
    """
    Create/update Viser geometry to match req_world exactly.
    """
    alive_keys = set()

    for i, obj in enumerate(req_world):
        typ = obj.get("type", "")
        key = f"{typ}_{i}"
        alive_keys.add(key)
        path = f"/world/{key}"

        if typ == "sphere":
            center = np.array(obj["center"], dtype=float)
            radius = float(obj["radius"])

            if key not in world_vis_handles:
                h = server.scene.add_transform_controls(
                    path,
                    position=tuple(center.tolist()),
                    scale=radius * 2.0,
                )
                server.scene.add_mesh_trimesh(
                    f"{path}/mesh",
                    mesh=sphere_template.to_trimesh(),
                )
                world_vis_handles[key] = h
            else:
                h = world_vis_handles[key]
                h.position = tuple(center.tolist())
                h.scale = radius * 2.0

        elif typ == "plane":
            # Plane visualization: draw a large grid once
            if key not in world_vis_handles:
                h = server.scene.add_grid(
                    path,
                    width=2.0,
                    height=2.0,
                    cell_size=0.1,
                )
                world_vis_handles[key] = h

    # Remove stale visuals
    for key in list(world_vis_handles.keys()):
        if key not in alive_keys:
            server.scene.remove_node(f"/world/{key}")
            del world_vis_handles[key]


# ============================================================
# Main
# ============================================================

def main():
    # ------------------------------------------------------------
    # Pyroki setup
    # ------------------------------------------------------------
    urdf_path = "/home/slurmlab/hao/pyronot/third_party/foam/assets/xarm7_leaphand/xarm7_allegro_right_spheres.urdf"
    urdf = yourdfpy.URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollisionSpherized.from_urdf(urdf)

    target_link_name = "link7"
    len_traj, dt = 5, 0.1
    prev_sols = np.array(
        robot.joint_var_cls.default_factory()[None].repeat(len_traj, axis=0)
    )

    total_dof = prev_sols.shape[-1]
    hand_dof = total_dof - ARM_DOF

    # ------------------------------------------------------------
    # World collision templates
    # ------------------------------------------------------------
    plane_coll = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    sphere_template = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.05]),
    )

    # ------------------------------------------------------------
    # Viser setup
    # ------------------------------------------------------------
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    init_wxyz = _quat_wxyz_from_rpy(np.array([3.14, 0.0, 0.0]))
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.2,
        position=(0.3, 0.0, 0.5),
        wxyz=init_wxyz,
    )

    timing_handle = server.gui.add_number("Solve time (ms)", 0.0, disabled=True)

    source_handle = server.gui.add_dropdown(
        "Target source",
        options=["viser (interactive)", "client request"],
        initial_value="client request",
    )

    # World visualization cache
    world_vis_handles: Dict[str, Any] = {}

    # ------------------------------------------------------------
    # ZeroMQ
    # ------------------------------------------------------------
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://127.0.0.1:5555")
    sock.setsockopt(zmq.RCVTIMEO, 100)

    print("[Pyroki] Planner+Viser server running at tcp://127.0.0.1:5555")

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------
    while True:
        try:
            req = sock.recv_pyobj()
        except zmq.error.Again:
            time.sleep(0.005)
            continue

        t0 = time.time()
        q = np.asarray(req["q"], dtype=float)

        if source_handle.value == "viser (interactive)":
            target_pos = np.array(ik_target_handle.position)
            target_quat = np.array(ik_target_handle.wxyz)
            hand_target = np.array(req.get("hand_target", []), dtype=float)

            world_coll = _make_world_coll(
                None,
                plane_coll,
                sphere_template,
            )

        else:
            target_pos = np.array(req["target_pos"], dtype=float)
            target_quat = np.array(req["target_quat"], dtype=float)
            hand_target = np.array(req.get("hand_target", []), dtype=float)

            req_world = req.get("world", [])
            world_coll = _make_world_coll(
                req_world,
                plane_coll,
                sphere_template,
            )

            # 🔴 VISUALIZE CLIENT WORLD 🔴
            update_world_visualization(
                server,
                req_world,
                sphere_template,
                world_vis_handles,
            )

            ik_target_handle.position = tuple(target_pos.tolist())
            ik_target_handle.wxyz = tuple(target_quat.tolist())

        # --------------------------
        # Solve
        # --------------------------
        sol_traj, sol_pos, sol_wxyz = pks.solve_online_planning_hand(
            robot=robot,
            robot_coll=robot_coll,
            world_coll=world_coll,
            target_link_name=target_link_name,
            target_position=target_pos,
            target_wxyz=target_quat,
            timesteps=len_traj,
            dt=dt,
            start_cfg=q,
            prev_sols=prev_sols,
            hand_target=hand_target,
        )

        prev_sols = sol_traj
        q_cmd = sol_traj[0]

        solve_ms = (time.time() - t0) * 1000.0
        timing_handle.value = 0.9 * timing_handle.value + 0.1 * solve_ms

        urdf_vis.update_cfg(q_cmd)

        sock.send_pyobj(
            {
                "q_cmd": q_cmd,
                "solve_ms": solve_ms,
                "status": "ok",
            }
        )


if __name__ == "__main__":
    main()
