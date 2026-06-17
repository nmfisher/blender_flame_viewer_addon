"""Update bone head/tail to track the deformed FLAME mesh.

Called every time ``update_mesh`` updates verts. Reads triplanar
anchor IDs stashed by ``bake_anchors`` and reconstructs each bone's
endpoints from the deformed mesh, briefly entering edit mode to modify
edit_bones.

Skips silently when:
    - no armature is selected in the scene
    - the armature has no anchors stashed
    - the user is currently editing the armature in edit mode
    - the FLAME_Viewer mesh isn't in the scene
"""

import json
import bpy
import numpy as np
from mathutils import Vector

MESH_NAME = "FLAME_Viewer"


def get_target_mesh(scene):
    """Return the user-picked target mesh for bone-anchor projection.

    Falls back to the auto-created FLAME_Viewer mesh if no explicit pick
    has been made (or the pick was cleared). Used by bake_anchors,
    update_bones, and anchor_markers — the same mesh handles both the
    bake-time projection and the per-frame coord lookups, so vertex IDs
    stay consistent.
    """
    obj = getattr(scene, "flame_viewer_target_mesh", None)
    if obj is not None and obj.type == "MESH":
        return obj
    return bpy.data.objects.get(MESH_NAME)


def _reconstruct(coords, ids):
    return (
        float(coords[ids["top"]][0]),
        float(coords[ids["right"]][1]),
        float(coords[ids["front"]][2]),
    )


def update_bones(context):
    scene = context.scene
    arm_obj = getattr(scene, "flame_viewer_armature", None)
    if arm_obj is None or arm_obj.type != "ARMATURE":
        return
    if "flame_viewer_anchors" not in arm_obj.keys():
        return
    if arm_obj.mode == "EDIT":
        return

    mesh_obj = get_target_mesh(scene)
    if mesh_obj is None:
        return

    try:
        anchors = json.loads(arm_obj["flame_viewer_anchors"])
    except (json.JSONDecodeError, TypeError, ValueError):
        return

    mesh = mesh_obj.data
    n_verts = len(mesh.vertices)
    coords = np.empty(n_verts * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape(n_verts, 3)

    reconstructed = {}
    for bone_name, sides in anchors.items():
        reconstructed[bone_name] = {
            "head": _reconstruct(coords, sides["head"]),
            "tail": _reconstruct(coords, sides["tail"]),
        }

    prev_active = context.view_layer.objects.active
    prev_mode = (arm_obj.mode if arm_obj == prev_active else "OBJECT")
    context.view_layer.objects.active = arm_obj
    try:
        bpy.ops.object.mode_set(mode="EDIT")
    except RuntimeError:
        return

    # Reconstructed positions are in mesh-local space; edit_bones.head/
    # tail expect armature-local. Convert via world to match the bake.
    arm_data = arm_obj.data
    mesh_world = mesh_obj.matrix_world
    arm_inv = arm_obj.matrix_world.inverted()
    for bone_name, recon in reconstructed.items():
        eb = arm_data.edit_bones.get(bone_name)
        if eb is None:
            continue
        eb.head = arm_inv @ (mesh_world @ Vector(recon["head"]))
        eb.tail = arm_inv @ (mesh_world @ Vector(recon["tail"]))

    target_mode = prev_mode if prev_mode in {"OBJECT", "POSE"} else "OBJECT"
    try:
        bpy.ops.object.mode_set(mode=target_mode)
    except RuntimeError:
        pass
    context.view_layer.objects.active = prev_active


class FLAMEVIEWER_OT_UpdateBones(bpy.types.Operator):
    bl_idname = "flameviewer.update_bones"
    bl_label = "Update Bones"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        update_bones(context)
        return {"FINISHED"}


def register():
    bpy.utils.register_class(FLAMEVIEWER_OT_UpdateBones)


def unregister():
    bpy.utils.unregister_class(FLAMEVIEWER_OT_UpdateBones)
