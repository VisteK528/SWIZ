import os
import random
import time


BASE_X = 0.0
BASE_Y = 0.62
BASE_Z = 1.2
MODEL_PATH = "/ws/install/swiz_binpicking/share/swiz_binpicking/worlds/ur/models/drc_practice_blue_cylinder/model.sdf"

for i in range(1, 20):
    x = BASE_X + random.uniform(-0.1, 0.1)
    y = BASE_Y + random.uniform(-0.1, 0.1)
    z = BASE_Z + (i * 0.15)
    
    r = random.uniform(0, 3.14)
    p = random.uniform(0, 3.14)
    y_rot = random.uniform(0, 3.14)
    
    name = f"blue_cylinder_{i}"
    
    req_string = (
        f"sdf_filename: '{MODEL_PATH}', "
        f"name: '{name}', "
        f"pose: {{ "
        f"  position: {{ x: {x:.3f}, y: {y:.3f}, z: {z:.3f} }}, "
        f"  orientation: {{ x: {r:.3f}, y: {p:.3f}, z: {y_rot:.3f} }} "
        f"}}"
    )
    
    cmd = f"gz service -s /world/default/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 300 --req \"{req_string}\""
    
    print(f"{name}: pos=[{x:.2f}, {y:.2f}, {z:.2f}] rot=[{r:.2f}, {p:.2f}, {y_rot:.2f}]")
    os.system(cmd)
    time.sleep(0.05)