import ai2thor.controller
import threading
import os,sys
import signal
import time
import psutil
from PIL import Image  # pip install Pillow

def kill_ai2thor():
    """Find and force-kill AI2-THOR and its Unity subprocesses."""
    for process in psutil.process_iter(attrs=['pid', 'name']):
        name = process.info["name"].lower()
        if "thor" in name or "unity" in name:
            print(f"Killing AI2-THOR process: {process.info['pid']} ({name})")
            try:
                os.kill(process.info['pid'], signal.SIGKILL)  # Kill process
            except Exception as e:
                print(f"Failed to kill {name}: {e}")

def reset_with_timeout(controller, scene_name, timeout=10):
    """Attempts to reset AI2-THOR with a timeout. If it hangs, force-kills everything."""
    def reset_controller():
        try:
            print(f"Attempting to reset AI2-THOR to {scene_name}...")
            controller.reset(scene_name=scene_name)
            print("Reset successful.")
        except Exception as e:
            print("Error during reset:", e)

    reset_thread = threading.Thread(target=reset_controller)
    reset_thread.start()
    reset_thread.join(timeout)  # Wait for reset to complete

    if reset_thread.is_alive():
        print("Reset is stuck! Killing AI2-THOR and Unity processes...")
        kill_ai2thor()  # Kill all AI2-THOR and Unity processes

        print("Stopping controller...")
        try:
            controller.stop()  # Try stopping AI2-THOR cleanly
        except Exception as e:
            print(f"Error stopping controller: {e}")

        time.sleep(2)  # Give time for processes to fully terminate
        print("Restarting AI2-THOR...")

        # Start a fresh AI2-THOR instance
        controller = ai2thor.controller.Controller()
        controller.reset(scene_name=scene_name)
        print("AI2-THOR restarted successfully.")

    return controller


def close_enough(obj_pos, agent_pos, threshold=0.2):
    dx = obj_pos["x"] - agent_pos["x"]
    dy = obj_pos["y"] - agent_pos["y"]
    dz = obj_pos["z"] - agent_pos["z"]
    return (dx*dx + dy*dy + dz*dz) ** 0.5 <= threshold

# Example: move camera to always be above the agent
def update_overhead_camera(controller):
    agent_metadata = controller.last_event.metadata["agent"]
    agent_pos = agent_metadata["position"]

    # Place camera slightly above agent
    new_camera_pos = {
        "x": agent_pos["x"],
        "y": agent_pos["y"] + 1.5,
        "z": agent_pos["z"]
    }

    # There's a dedicated action in newer AI2-THOR versions:
    event = controller.step(
        action="UpdateThirdPartyCamera",
        thirdPartyCameraId=0,  # If you only added one camera
        position=new_camera_pos,
        rotation={"x": 90, "y": 0, "z": 0},
        fieldOfView=90,
        orthographic=False
    )

    if not event.metadata["lastActionSuccess"]:
        print("Camera update failed:", event.metadata["errorMessage"])


# Initialize AI2-THOR
# controller = ai2thor.controller.Controller( width=640, height=480)
controller = ai2thor.controller.Controller()
time.sleep(2)

# Reset and load the scene
# controller.reset(scene_name="FloorPlan1")
# Use the safe reset function
controller = reset_with_timeout(controller, "FloorPlan1")

# Add a third-party camera above the scene, pointed straight down (x=90°)
controller.step(
    action="AddThirdPartyCamera",
    position={"x": 0, "y": 1.5, "z": 0},  # Adjust Y higher or lower if needed
    rotation={"x": 90, "y": 0, "z": 0},
    fieldOfView=90
)




# Ensure AI2-THOR is running properly
event = controller.step(action="Pass")
if isinstance(event, str):
    print("Error:", event)  # Debugging output
else:
    print("AI2-THOR initialized successfully.")

allowed_actions = event.metadata["actionReturn"]

# Move the agent
controller.step(action="MoveAhead")
allowed_actions = event.metadata["actionReturn"]
print("allowed_actions",allowed_actions)

# Get all valid object types in the current scene
valid_objects = {obj["objectType"] for obj in controller.last_event.metadata["objects"]}
print("Valid object types in this scene:", valid_objects)

# Spawn an object (e.g., Apple)
event = controller.step(
    action="CreateObject",
    objectType="Apple",
    position={"x": 0.5, "y": 1.0, "z": -1.2},
    forceAction=True
)
# Get the spawned object's ID
if event.metadata["lastActionSuccess"]:
    apple_id = event.metadata["actionReturn"]
    print("Apple created with ID:", apple_id)

    # Modify the object to simulate a new type
    controller.step(
        action="ChangeColor",
        objectId=apple_id,
        r=0, g=0, b=255  # Change to blue
    )
    print("Apple color changed to blue (simulating a new object type).")
else:
    print("Error:", event.metadata["errorMessage"])


event = controller.step(action="GetReachablePositions")
scene_objects = controller.last_event.metadata["objects"]

def dist(obj_pos, agent_pos):
    dx = obj_pos["x"] - agent_pos["x"]
    dy = obj_pos["y"] - agent_pos["y"]
    dz = obj_pos["z"] - agent_pos["z"]
    return ((dx*dx + dy*dy + dz*dz)**0.5)
def nameToID( name):
    global controller
    oid = None
    scene_objects = controller.last_event.metadata["objects"]
    for obj in scene_objects:
         if name == obj['name']: oid = obj["objectId"]
    return( oid )
def listPickupable():
    global controller
    oid = None
    scene_objects = controller.last_event.metadata["objects"]
    for obj in scene_objects:
         if obj['pickupable']: print(f"{obj['name']} can be picked up ID = {obj['objectId']}" )

print("All objects")
metadata = controller.last_event.metadata
agent_position = metadata["agent"]["position"]

for obj in scene_objects:
    # print(f"    Object: {obj['name']}, Position: {obj['position']} Distance: { dist(obj['position'], agent_position) }")
    print(f"{obj['name']}, {obj['objectType']} {obj['objectId']}" ) # 'objectType', 'objectId'
print(f"{obj.keys() }")

# Locations for all objects
# for obj in scene_objects:
#     if (dist(obj['position'], agent_position))<0.3: print(f"{obj['name']} is here ({ dist(obj['position'], agent_position) })")

help =""" Allowed motions and actions:
    MoveAhead MoveBack MoveLeft MoveRight RotateLeft RotateRight
    LookUp LookDown Teleport TeleportFull Pass PickupObject DropHandObject
    ThrowObject PutObject OpenObject CloseObject ToggleObjectOn ToggleObjectOff
    SliceObject BreakObject DirtyObject CleanObject
    FillObjectWithLiquid EmptyLiquidFromObject UseUpObject CreateObject DestroyObject
    NontThor commands: pick, left, right, move, back, refresh
    """
allowedCommands = [
    "MoveAhead","MoveBack","MoveLeft","MoveRight","RotateLeft","RotateRight",
    "LookUp","LookDown","Teleport","TeleportFull","Pass","PickupObject","DropHandObject",
    "ThrowObject","PutObject","OpenObject","CloseObject","ToggleObjectOn","ToggleObjectOff",
    "SliceObject","BreakObject","DirtyObject","CleanObject",
    "FillObjectWithLiquid","EmptyLiquidFromObject","UseUpObject","CreateObject","DestroyObject",
    "pick,","left,","right,","move,","back,","refresh",
    ]
lcCommands = [s.lower() for s in allowedCommands]

print(help)
closebyObjects = []
while True:
   s = sys.stdin.readline().strip()

   if (s not in allowedCommands) and (s.lower() in lcCommands):
      print("wrong capitalization")
      print( allowedCommands[ lcCommands.index( s ) ] )
      s = allowedCommands[ lcCommands.index( s ) ]
   if s=="pick": s="PickupObject"
   if s=="left": s="RotateLeft"
   if s=="right": s="RotateRight"
   if s=="move": s="MoveAhead"
   if s=="back": s="MoveBack"
   if s=="refresh":
       for obj in scene_objects:
        if (dist(obj['position'], agent_position))<0.5:
            print(f"{obj['name']} is here ({ dist(obj['position'], agent_position) })")
            if obj['pickupable']: print(f" can be picked up")
            if obj['visible']: print(f" and is visible")
       inventory = controller.last_event.metadata["inventoryObjects"]
       if inventory:
            print("You're currently holding:")
            for obj in inventory:
                print(f" - objectId: {obj['objectId']}, objectType: {obj['objectType']}")
       continue

   if s=="?":
       print(help)
       continue
   try:
       if "," in s:
           event = eval( 'controller.step('+s+')' )
       else:
           if "Object" in s:
               print("Act on what objectID or object name:")
               o = sys.stdin.readline().strip()
               if not '|' in o: o = nameToID(o) # assume pipe symbols are in IDs and not names
               event = controller.step( action=s, objectId=o )
           else:
               event = controller.step( action=s )
   except ValueError:
       print("Bad action")
       continue

   if not event.metadata["lastActionSuccess"]: print("Error:", event.metadata["errorMessage"])
   event = controller.step(action="Pass")
   update_overhead_camera(controller)

   # The overhead camera image is in event.third_party_camera_frames[0]
   overhead_view = event.third_party_camera_frames[0]
   img = Image.fromarray(overhead_view, 'RGB')
   img.show()

   if event.metadata["actionReturn"]: print("Special allowed Actions:", event.metadata["actionReturn"] )

   metadata = controller.last_event.metadata
   agent_position = metadata["agent"]["position"]
   print( "Agent position",agent_position )

   for obj in scene_objects:
    if obj['name'] == "Agent": continue
    if (dist(obj['position'], agent_position))<1.4:  # within 1.4 meters
        closebyObjects.append(  obj['name'] )
        if obj['name'] in closebyObjects:
            print(f"  {obj['name']} is here ({ dist(obj['position'], agent_position) })", end=" ")
            if obj['visible']: print(f" and is visible", end=" ")
            if obj['pickupable']: print(f" and is the kind of object that can be picked up", end=" ")
            print("")
    else:
        # remove from closebyObjects if it's not close
        try: closebyObjects.remove( obj['name'] )
        except ValueError: pass

   inventory = controller.last_event.metadata["inventoryObjects"]
   if inventory:
        print("You're currently holding:")
        for obj in inventory:
            print(f"   - objectId: {obj['objectId']}, objectType: {obj['objectType']}")