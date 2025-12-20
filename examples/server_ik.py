#!/usr/bin/env python3
"""
ZMQ-based IK server using Pyroki.
Receives a sequence of end-effector poses and returns joint trajectories.

Request:
{
  "eef_pose": (T, 6)  # xyz + rpy (xyz Euler)
}

Response:
{
  "q": (T, dof)
}
"""

import zmq
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

import pyronot as pk
import pyronot_snippets as pks
import yourdfpy


ARM_DOF = 7  # xArm7


# -----------------------------
# Utils
# -----------------------------

def quat_wxyz_from_rpy(rpy_xyz: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_euler("xyz", rpy_xyz).as_quat()
    return np.roll(quat_xyzw, 1)  # xyzw -> wxyz


# -----------------------------
# IK Server
# -----------------------------

class PyrokiIKServer:
    def __init__(
        self,
        urdf_path: str,
        target_link_name: str,
        bind_addr: str = "tcp://*:5555",
    ):
        print("[IK-SERVER] Loading URDF...")
        urdf = yourdfpy.URDF.load(urdf_path)

        self.robot = pk.Robot.from_urdf(urdf)
        self.target_link_name = target_link_name

        # ZMQ
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.bind(bind_addr)

        print(f"[IK-SERVER] Listening on {bind_addr}")

    def solve_sequence(self, eef_pose_seq: np.ndarray) -> np.ndarray:
        """
        eef_pose_seq: (T,6) xyz + rpy
        returns q_seq: (T,dof)
        """
        T = eef_pose_seq.shape[0]
        q_out = np.zeros((T, ARM_DOF), dtype=np.float32)

        for t in range(T):
            pos = eef_pose_seq[t, :3]
            rpy = eef_pose_seq[t, 3:6]
            quat = quat_wxyz_from_rpy(rpy)

            try:
                q = pks.solve_ik(
                    robot=self.robot,
                    target_link_name=self.target_link_name,
                    target_position=pos,
                    target_wxyz=quat,
                )
            except Exception as e:
                print(f"[IK-SERVER] IK failed at t={t}: {e}")

            q_out[t] = q[:ARM_DOF]

        return q_out

    def serve_forever(self):
        while True:
            msg = self.sock.recv()
            req = json.loads(msg.decode("utf-8"))

            eef_pose = np.asarray(req["eef_pose"], dtype=np.float32)
            assert eef_pose.ndim == 2 and eef_pose.shape[1] == 6

            q = self.solve_sequence(eef_pose)

            resp = {
                "q": q.tolist(),
            }
            self.sock.send(json.dumps(resp).encode("utf-8"))


# -----------------------------
# Entry
# -----------------------------

def main():
    urdf_path = "/home/slurmlab/hao/pyronot/third_party/foam/assets/xarm7_leaphand/xarm7_allegro_right_spheres.urdf"
    target_link_name = "link7"

    server = PyrokiIKServer(
        urdf_path=urdf_path,
        target_link_name=target_link_name,
        bind_addr="tcp://*:5555",
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
