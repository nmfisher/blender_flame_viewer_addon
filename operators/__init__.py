from . import (load_model, update_bones, update_mesh, randomize,
               bake_anchors, anchor_markers, debug_pick)


def register():
    load_model.register()
    update_bones.register()
    update_mesh.register()
    randomize.register()
    bake_anchors.register()
    anchor_markers.register()
    debug_pick.register()


def unregister():
    debug_pick.unregister()
    anchor_markers.unregister()
    bake_anchors.unregister()
    randomize.unregister()
    update_mesh.unregister()
    update_bones.unregister()
    load_model.unregister()
