"""Load a FLAME model pickle and create a mesh in the viewport."""

import os
import bpy

from .. import flame_numpy

MESH_NAME = "FLAME_Viewer"


# Module-level cache: survives between operator calls within a session.
_flame_cache = None
_cache_path = None


def get_cache():
    """Return (model_data, cache_path) or (None, None) if not loaded."""
    return _flame_cache, _cache_path


def ensure_cache(context):
    """Return (model_data, cache_path), reloading from the scene's stored
    pkl path if the in-memory cache was lost (e.g. after a Blender restart).
    Returns (None, None) if no path is stored or the file is missing.
    """
    global _flame_cache, _cache_path
    if _flame_cache is not None:
        return _flame_cache, _cache_path
    path = context.scene.flame_viewer_model_path
    if not path or not os.path.isfile(path):
        return None, None
    try:
        _flame_cache = flame_numpy.load_flame_model(path)
        _cache_path = path
    except Exception:
        return None, None
    return _flame_cache, _cache_path


def _flame_to_blender_axes(verts):
    """FLAME Y-up → Blender Z-up: (x, y, z) → (x, -z, y)."""
    out = verts.copy()
    out[:, 1] = -verts[:, 2]
    out[:, 2] = verts[:, 1]
    return out


def _blender_to_flame_axes(verts):
    """Blender Z-up → FLAME Y-up: (x, y, z) → (x, z, -y)."""
    out = verts.copy()
    out[:, 1] = verts[:, 2]
    out[:, 2] = -verts[:, 1]
    return out


class FLAMEVIEWER_OT_LoadModel(bpy.types.Operator):
    bl_idname = "flameviewer.load_model"
    bl_label = "Load FLAME PKL"
    bl_description = "Load a FLAME model pickle file"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(
        default="*.pkl", options={"HIDDEN"})

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = os.path.expanduser("~")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        global _flame_cache, _cache_path

        path = bpy.path.abspath(self.filepath)
        if not os.path.isfile(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}

        try:
            model = flame_numpy.load_flame_model(path)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to load PKL: {e}")
            return {"CANCELLED"}

        _flame_cache = model
        _cache_path = path

        n_verts = model["v_template"].shape[0]
        n_faces = model["f"].shape[0]

        # Convert to Blender axes
        v_blender = _flame_to_blender_axes(model["v_template"])

        # Remove old mesh if it exists
        old = bpy.data.objects.get(MESH_NAME)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)
        old_mesh = bpy.data.meshes.get(MESH_NAME)
        if old_mesh:
            bpy.data.meshes.remove(old_mesh)

        # Create mesh
        mesh = bpy.data.meshes.new(MESH_NAME)
        mesh.from_pydata(v_blender.tolist(), [], model["f"].tolist())
        mesh.update()

        obj = bpy.data.objects.new(MESH_NAME, mesh)
        context.scene.collection.objects.link(obj)

        # Select and make active
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        context.view_layer.objects.active = obj

        # Zoom to fit — find a 3D viewport space and call view_all
        for area in context.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    try:
                        with context.temp_override(
                            area=area,
                            region=area.regions[-1],
                            space_data=space,
                        ):
                            bpy.ops.view3d.view_all(center=True)
                    except RuntimeError:
                        pass
                    break
            break

        # Update scene properties
        scene = context.scene
        scene.flame_viewer_loaded = True
        scene.flame_viewer_model_path = path
        scene.flame_viewer_vert_count = n_verts
        scene.flame_viewer_face_count = n_faces

        # Reset sliders + overflow
        scene["flame_viewer_shape_overflow"] = [0.0] * 290
        scene["flame_viewer_expr_overflow"] = [0.0] * 90
        scene.flame_viewer_shape_params = [0.0] * 10
        scene.flame_viewer_expr_params = [0.0] * 10
        scene.flame_viewer_global = [0.0] * 3
        scene.flame_viewer_neck = [0.0] * 3
        scene.flame_viewer_jaw = [0.0] * 3
        scene.flame_viewer_eye_l = [0.0] * 3
        scene.flame_viewer_eye_r = [0.0] * 3

        self.report({"INFO"},
                     f"Loaded {os.path.basename(path)}: "
                     f"{n_verts} verts, {n_faces} faces")
        return {"FINISHED"}


def register():
    bpy.utils.register_class(FLAMEVIEWER_OT_LoadModel)


def unregister():
    bpy.utils.unregister_class(FLAMEVIEWER_OT_LoadModel)
