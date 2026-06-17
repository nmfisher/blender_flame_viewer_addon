"""Randomize and reset FLAME parameters.

Mirrors the rigatoni-style randomization UI: a sigma operator property,
three scene-level toggles (Shape 1-100, Shape 100-300, Expression), and
randomization across the full 400-dim FLAME coefficient space.

The visible sliders only cover the first 10 indices of shape and expression;
the overflow (shape 10-299, expr 10-99) lives in IDProperty arrays on the
scene so it persists with the .blend file.
"""

import numpy as np
import bpy

from .load_model import ensure_cache
from .update_mesh import (update_mesh, SHAPE_OVERFLOW_KEY, EXPR_OVERFLOW_KEY,
                          SHAPE_OVERFLOW_LEN, EXPR_OVERFLOW_LEN)


class FLAMEVIEWER_OT_Randomize(bpy.types.Operator):
    bl_idname = "flameviewer.randomize"
    bl_label = "Randomize"
    bl_description = "Randomize FLAME shape and expression parameters"
    bl_options = {"REGISTER", "UNDO"}

    sigma: bpy.props.FloatProperty(
        name="Sigma",
        description=("Standard deviation for random coefficients "
                     "(higher = more variation)"),
        default=1.5,
        min=0.1,
        max=5.0,
    )

    def execute(self, context):
        scene = context.scene

        model, _ = ensure_cache(context)
        if model is None:
            self.report(
                {"ERROR"},
                "FLAME model not loaded. Re-load the PKL "
                f"(saved path: {scene.flame_viewer_model_path or 'none'}).")
            return {"CANCELLED"}

        rng = np.random.default_rng()
        coeffs = rng.normal(0.0, self.sigma, size=400)
        if not scene.flame_viewer_randomize_shape_1:
            coeffs[:100] = 0.0
        if not scene.flame_viewer_randomize_shape_2:
            coeffs[100:300] = 0.0
        if not scene.flame_viewer_randomize_expression:
            coeffs[300:] = 0.0

        shape = coeffs[:300]
        expr = coeffs[300:]

        # Overflow first (no callback) so the slider update fires once with
        # the full state already in place.
        scene[SHAPE_OVERFLOW_KEY] = shape[10:].tolist()
        scene[EXPR_OVERFLOW_KEY] = expr[10:].tolist()

        # Visible sliders (clamped to slider range so the UI stays valid).
        scene.flame_viewer_shape_params = np.clip(
            shape[:10], -3.0, 3.0).tolist()
        scene.flame_viewer_expr_params = np.clip(
            expr[:10], -3.0, 3.0).tolist()

        # Pose toggles: if off, leave the slider's current value alone.
        if scene.flame_viewer_randomize_global_rotation:
            global_rot = np.clip(rng.normal(0.0, 0.2, size=3), -0.5, 0.5)
            scene.flame_viewer_global = global_rot.tolist()

        if scene.flame_viewer_randomize_jaw_pose:
            jaw = rng.normal(0.0, 0.15, size=3)
            jaw[0] = abs(jaw[0])  # jaw only opens
            scene.flame_viewer_jaw = np.clip(jaw, -1.0, 1.0).tolist()

        if scene.flame_viewer_randomize_eye_pose:
            eye_l = np.clip(rng.normal(0.0, 0.1, size=3), -1.0, 1.0)
            eye_r = np.clip(rng.normal(0.0, 0.1, size=3), -1.0, 1.0)
            scene.flame_viewer_eye_l = eye_l.tolist()
            scene.flame_viewer_eye_r = eye_r.tolist()

        return {"FINISHED"}


class FLAMEVIEWER_OT_Reset(bpy.types.Operator):
    bl_idname = "flameviewer.reset"
    bl_label = "Reset"
    bl_description = "Reset all parameters to neutral"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        scene[SHAPE_OVERFLOW_KEY] = [0.0] * SHAPE_OVERFLOW_LEN
        scene[EXPR_OVERFLOW_KEY] = [0.0] * EXPR_OVERFLOW_LEN
        scene.flame_viewer_shape_params = [0.0] * 10
        scene.flame_viewer_expr_params = [0.0] * 10
        scene.flame_viewer_global = [0.0] * 3
        scene.flame_viewer_neck = [0.0] * 3
        scene.flame_viewer_jaw = [0.0] * 3
        scene.flame_viewer_eye_l = [0.0] * 3
        scene.flame_viewer_eye_r = [0.0] * 3

        update_mesh(context)
        return {"FINISHED"}


def register():
    bpy.utils.register_class(FLAMEVIEWER_OT_Randomize)
    bpy.utils.register_class(FLAMEVIEWER_OT_Reset)


def unregister():
    bpy.utils.unregister_class(FLAMEVIEWER_OT_Reset)
    bpy.utils.unregister_class(FLAMEVIEWER_OT_Randomize)
