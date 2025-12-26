import viser
import yourdfpy
import pyronot as pk

from pyronot.collision import RobotCollisionSpherized
import numpy as np

# xArm7 (7 DOF) + Allegro (16 DOF) = 23 DOF total
q = np.array([
    # ---- arm ----
    0.0036,-0.4052,-0.0035,0.3886,0.01066,0.7939,0.0064,
    # ---- hand (Allegro) ----
     0.3, 0.6, 0.6, 0.3,      # index
     0.3, 0.6, 0.6, 0.3,      # middle
     0.3, 0.6, 0.6, 0.3,      # ring
     0.2, 0.4, 0.4, 0.2,      # thumb
], dtype=float)

# ----------------------------
# Load robot
# ----------------------------
urdf_path = "/home/slurmlab/hao/pyronot/third_party/foam/assets/xarm7_leaphand/xarm7_leap_right_spheres.urdf"
urdf = yourdfpy.URDF.load(urdf_path)

robot = pk.Robot.from_urdf(urdf)
robot_coll = RobotCollisionSpherized.from_urdf(urdf)

# ----------------------------
# Viser setup
# ----------------------------
server = viser.ViserServer()
server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

# ----------------------------
# Compute & visualize collision mesh
# ----------------------------
coll_geom_world = robot_coll.at_config(robot, q)

# IMPORTANT: this merges all link capsules into one mesh
coll_mesh = coll_geom_world.to_trimesh()
coll_mesh.visual.vertex_colors = [0, 180, 255, 140]  # cyan

server.scene.add_mesh_trimesh(
    "/robot_collision",
    mesh=coll_mesh,
    visible=True,
)

print("Collision geometry visualized. Open Viser to inspect.")

while True:
    pass
