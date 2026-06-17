"""Compute and stash triplanar anchor vertex IDs for an armature's bones.

For each bone endpoint (head and tail), find 3 FLAME mesh vertex IDs —
one per axis-aligned ortho view (front/+Y, left/+X, top/+Z) — that
reconstruct the endpoint position when the mesh deforms.

Run at canonical pose (all sliders zero). Anchors are stored as JSON on
the armature object's ``flame_viewer_anchors`` custom property; the
runtime side lives in ``update_bones``.
"""

import json
import bpy
import numpy as np
from mathutils import Vector

from .update_bones import get_target_mesh

MESH_NAME = "FLAME_Viewer"
EPSILON = 0.3


def _slider_zero(scene):
    for prop in ("flame_viewer_shape_params",
                 "flame_viewer_expr_params",
                 "flame_viewer_global",
                 "flame_viewer_neck",
                 "flame_viewer_jaw",
                 "flame_viewer_eye_l",
                 "flame_viewer_eye_r"):
        if any(abs(v) > 1e-6 for v in getattr(scene, prop)):
            return False
    for key in ("flame_viewer_shape_overflow", "flame_viewer_expr_overflow"):
        raw = scene.get(key)
        if raw is None:
            continue
        if any(abs(v) > 1e-6 for v in raw):
            return False
    return True


def find_anchors(coords, normals, p):
    """Return {'front': vid, 'right': vid, 'top': vid} for endpoint p.

    coords:  (V, 3) Blender Z-up mesh coords.
    normals: (V, 3) per-vertex normals in the same frame.
    p:       (3,) endpoint position.

    For each ortho view, we mask vertices to those whose normal points
    toward the camera (n . view_dir < -EPSILON), then pick the one
    whose 2D in-plane projection is closest to p's projection.

    Cameras (matching the user's stated convention):
        front: camera at  -Y, looking +Y; image plane (X, Z)
        right: camera at  +X, looking -X; image plane (Y, Z)
        top:   camera at  +Z, looking -Z; image plane (X, Y)
    """
    p = np.asarray(p, dtype=np.float64)

    # (look_direction, in-plane axes a, b)
    views = {
        "front": (np.array([ 0.,  1.,  0.]), 0, 2),
        "right": (np.array([-1.,  0.,  0.]), 1, 2),
        "top":   (np.array([ 0.,  0., -1.]), 0, 1),
    }

    out = {}
    for name, (view_dir, a, b) in views.items():
        dots = normals @ view_dir
        visible = dots < -EPSILON
        if not visible.any():
            visible = np.ones(len(coords), dtype=bool)
        da = coords[:, a] - p[a]
        db = coords[:, b] - p[b]
        d2 = da * da + db * db
        d2_masked = np.where(visible, d2, np.inf)
        out[name] = int(np.argmin(d2_masked))
    return out


def reconstruct(coords, ids):
    """Reconstruct an endpoint from {'front', 'right', 'top'} vertex IDs."""
    return (
        float(coords[ids["top"]][0]),    # X — in-plane in top view
        float(coords[ids["right"]][1]),  # Y — in-plane in right view
        float(coords[ids["front"]][2]),  # Z — in-plane in front view
    )


class FLAMEVIEWER_OT_BakeAnchors(bpy.types.Operator):
    bl_idname = "flameviewer.bake_anchors"
    bl_label = "Bake Anchors"
    bl_description = ("Compute triplanar anchor vertex IDs for the "
                      "selected armature's bones. Run at canonical pose "
                      "(all sliders zero).")

    @classmethod
    def poll(cls, context):
        scene = context.scene
        arm = getattr(scene, "flame_viewer_armature", None)
        return (arm is not None
                and arm.type == "ARMATURE"
                and get_target_mesh(scene) is not None)

    def execute(self, context):
        scene = context.scene
        arm_obj = scene.flame_viewer_armature
        mesh_obj = get_target_mesh(scene)

        if not _slider_zero(scene):
            self.report({"ERROR"},
                        "All FLAME parameters must be at zero "
                        "(canonical pose) when baking anchors.")
            return {"CANCELLED"}

        if not arm_obj.data.bones:
            self.report({"ERROR"}, f"Armature '{arm_obj.name}' has no bones.")
            return {"CANCELLED"}

        mesh = mesh_obj.data
        n_verts = len(mesh.vertices)
        coords = np.empty(n_verts * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", coords)
        coords = coords.reshape(n_verts, 3)

        normals = np.empty(n_verts * 3, dtype=np.float64)
        mesh.vertices.foreach_get("normal", normals)
        normals = normals.reshape(n_verts, 3)

        # bone.head_local / tail_local are in armature-object-local
        # space; mesh.vertices.co is in mesh-object-local space. Route
        # through world space so the bake works regardless of either
        # object's world transform (matching the debug-pick path).
        arm_world = arm_obj.matrix_world
        mesh_inv = mesh_obj.matrix_world.inverted()

        anchors = {}
        max_err = 0.0
        for bone in arm_obj.data.bones:
            head = tuple(mesh_inv @ (arm_world @ Vector(bone.head_local)))
            tail = tuple(mesh_inv @ (arm_world @ Vector(bone.tail_local)))
            head_ids = find_anchors(coords, normals, head)
            tail_ids = find_anchors(coords, normals, tail)
            anchors[bone.name] = {"head": head_ids, "tail": tail_ids}

            head_recon = np.array(reconstruct(coords, head_ids))
            tail_recon = np.array(reconstruct(coords, tail_ids))
            err = max(np.linalg.norm(head_recon - np.array(head)),
                      np.linalg.norm(tail_recon - np.array(tail)))
            max_err = max(max_err, err)

        arm_obj["flame_viewer_anchors"] = json.dumps(anchors)

        # Stale markers carry the previous vids — drop them so the user
        # has to re-click "Show Markers" and gets a fresh set.
        from .anchor_markers import clear_markers
        clear_markers()

        max_err_mm = max_err * 1000.0
        scene.flame_viewer_anchor_status = (
            f"baked: {len(anchors)} bones, "
            f"max error {max_err_mm:.1f} mm")
        self.report({"INFO"},
                    f"Baked {len(anchors)} bones, "
                    f"max anchor error {max_err_mm:.1f} mm")
        return {"FINISHED"}


def register():
    bpy.utils.register_class(FLAMEVIEWER_OT_BakeAnchors)


def unregister():
    bpy.utils.unregister_class(FLAMEVIEWER_OT_BakeAnchors)
