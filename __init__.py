bl_info = {
    "name": "FLAME Viewer",
    "author": "metahuman_automation",
    "version": (0, 1, 0),
    "blender": (4, 3, 0),
    "location": "View3D > Sidebar > FLAME Viewer",
    "description": "Load FLAME models and interactively explore shape/expression parameters",
    "category": "3D View",
}


def register():
    from . import panels, operators
    panels.register()
    operators.register()


def unregister():
    from . import operators, panels
    operators.unregister()
    panels.unregister()
