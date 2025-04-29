from ai2thor.controller import Controller

controller = Controller()
controller.start()

controller.reset('FloorPlan1')
event = controller.step(dict(action='Initialize', gridSize=0.25))
objects = event.metadata['objects']

type_id_list = [(o['objectType'], o['objectId']) for o in objects]

for obj_type, obj_id in type_id_list:
    print(f"{obj_type:20s} -- {obj_id}")