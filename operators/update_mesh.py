"""Update FLAME mesh vertices from current slider values."""

import numpy as np
import bpy

from .. import flame_numpy
from .load_model import ensure_cache, _flame_to_blender_axes
from .update_bones import update_bones
from .anchor_markers import update_markers

MESH_NAME = "FLAME_Viewer"

# Overflow storage for the indices the visible 10-slider UI doesn't cover.
# Lives as IDProperty arrays on the scene so it persists across save/reload.
SHAPE_OVERFLOW_KEY = "flame_viewer_shape_overflow"
EXPR_OVERFLOW_KEY = "flame_viewer_expr_overflow"
SHAPE_OVERFLOW_LEN = 290  # indices 10..299
EXPR_OVERFLOW_LEN = 90    # indices 10..99


def _read_overflow(scene, key, length):
    raw = scene.get(key)
    out = np.zeros(length, dtype=np.float64)
    if raw is None:
        return out
    arr = np.asarray(list(raw), dtype=np.float64)
    n = min(arr.size, length)
    out[:n] = arr[:n]
    return out


def update_mesh(context):
    """Re-evaluate FLAME model and push vertices to the mesh object."""
    model, _ = ensure_cache(context)
    if model is None:
        return

    obj = bpy.data.objects.get(MESH_NAME)
    if obj is None:
        return

    scene = context.scene

    # Combine visible sliders (first 10) with the IDProperty overflow.
    shape_full = np.zeros(300, dtype=np.float64)
    expr_full = np.zeros(100, dtype=np.float64)

    sp = scene.flame_viewer_shape_params
    for i in range(min(10, len(sp))):
        shape_full[i] = sp[i]
    shape_full[10:] = _read_overflow(scene, SHAPE_OVERFLOW_KEY,
                                     SHAPE_OVERFLOW_LEN)

    ep = scene.flame_viewer_expr_params
    for i in range(min(10, len(ep))):
        expr_full[i] = ep[i]
    expr_full[10:] = _read_overflow(scene, EXPR_OVERFLOW_KEY,
                                    EXPR_OVERFLOW_LEN)

    jaw = np.array(list(scene.flame_viewer_jaw), dtype=np.float64)
    neck = np.array(list(scene.flame_viewer_neck), dtype=np.float64)
    global_rot = np.array(list(scene.flame_viewer_global), dtype=np.float64)
    eye_l = np.array(list(scene.flame_viewer_eye_l), dtype=np.float64)
    eye_r = np.array(list(scene.flame_viewer_eye_r), dtype=np.float64)

    transl = np.zeros(3, dtype=np.float64)

    pose = np.zeros(6, dtype=np.float64)
    pose[:3] = global_rot
    pose[3:] = jaw
    eye_pose = np.concatenate([eye_l, eye_r])

    verts = flame_numpy.flame_forward(
        shape_full, expr_full, pose, neck, transl, model, eye_pose=eye_pose)

    # Convert to Blender axes
    verts_blender = _flame_to_blender_axes(verts)

    # Push to mesh
    mesh = obj.data
    flat = verts_blender.astype(np.float32).ravel()
    mesh.vertices.foreach_set("co", flat)
    mesh.update()

    update_bones(context)
    update_markers(context)


class FLAMEVIEWER_OT_UpdateMesh(bpy.types.Operator):
    bl_idname = "flameviewer.update_mesh"
    bl_label = "Update FLAME Mesh"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        update_mesh(context)
        return {"FINISHED"}


def _on_slider_change(self, context):
    """Callback for slider property updates."""
    update_mesh(context)


def register():
    bpy.utils.register_class(FLAMEVIEWER_OT_UpdateMesh)


def unregister():
    bpy.utils.unregister_class(FLAMEVIEWER_OT_UpdateMesh)
