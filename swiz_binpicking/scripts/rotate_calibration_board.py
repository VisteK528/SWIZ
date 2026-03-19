import os
import random
import math

MODEL_NAME = "checkerboard_7_9_0_02"
WORLD_NAME = "default"

BASE_X, BASE_Y, BASE_Z = 0.0, 0.6, 0.6

POS_VAR = 0.05  
ROT_VAR = math.radians(2)

def move():
    x = BASE_X + random.uniform(-POS_VAR, POS_VAR)
    y = BASE_Y + random.uniform(-POS_VAR, POS_VAR)
    z = BASE_Z 
    
    r = random.uniform(-ROT_VAR, ROT_VAR)
    p = random.uniform(-ROT_VAR, ROT_VAR)
    y_rot = random.uniform(-ROT_VAR, ROT_VAR)

    req_string = (
        f"name: '{MODEL_NAME}', "
        f"position: {{ x: {x:.4f}, y: {y:.4f}, z: {z:.4f} }}, "
        f"orientation: {{ x: {r:.4f}, y: {p:.4f}, z: {y_rot:.4f} }}"
    )

    cmd = (
        f"gz service -s /world/{WORLD_NAME}/set_pose "
        f"--reqtype gz.msgs.Pose "
        f"--reptype gz.msgs.Boolean "
        f"--timeout 300 "
        f"--req \"{req_string}\""
    )

    os.system(cmd)

if __name__ == "__main__":
    move()