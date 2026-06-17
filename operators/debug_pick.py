"""Debug pick: project a chosen bone's head and tail from the current
3D viewport view direction onto the target mesh, and drop colored
markers at the picked vertices.

Unlike ``bake_anchors`` (which works in object-local coords and assumes
both armature and mesh sit at world origin), this operator goes through
world space — so when the bake reports a huge error, this can reveal
whether the bones are simply far from the mesh in world coords (e.g.,
appended-armature world transform mismatch) versus an in-frame issue.
"""

import bpy
import numpy as np
from mathutils import Vector

from .update_bones import get_target_mesh

EPSILON = 0.3
DEBUG_COLL_NAME = "FLAMEViewer_DebugPicks"
DEBUG_PREFIX = "FLAMEDebugPick_"
DEBUG_MARKER_SIZE = 0.005

DEBUG_COLORS = {
    "head": (1.0, 0.1, 0.9, 1.0),  # magenta
    "tail": (0.1, 0.9, 1.0, 1.0),  # cyan
}

_bone_items_cache = []


def _bone_enum_items(self, context):
    """Populate the bone dropdown from the currently picked armature."""
    global _bone_items_cache
    if context is None:
        _bone_items_cache = [("__none__", "(none)", "")]
        return _bone_items_cache
    arm = getattr(context.scene, "flame_viewer_armature", None)
    if arm is None or arm.type != "ARMATURE":
        _bone_items_cache = [("__none__", "(pick an armature)", "")]
        return _bone_items_cache
    bones = arm.data.bones
    if not bones:
        _bone_items_cache = [("__none__", "(no bones)", "")]
        return _bone_items_cache
    _bone_items_cache = [(b.name, b.name, "") for b in bones]
    return _bone_items_cache


def _ensure_collection():
    coll = bpy.data.collections.get(DEBUG_COLL_NAME)
    if coll is None:
        coll = bpy.data.collections.new(DEBUG_COLL_NAME)
        bpy.context.scene.collection.children.link(coll)
    return coll


def _clear():
    coll = bpy.data.collections.get(DEBUG_COLL_NAME)
    if coll is None:
        return
    for obj in list(coll.objects):
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    bpy.data.collections.remove(coll)


def _ensure_material(role):
    name = f"FLAMEDebug_Mat_{role}"
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name=name)
    mat.diffuse_color = DEBUG_COLORS[role]
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = DEBUG_COLORS[role]
    em.inputs["Strength"].default_value = 4.0
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def _make_marker(name, role, location, coll):
    s = DEBUG_MARKER_SIZE
    verts = [(-s, -s, -s), ( s, -s, -s), ( s,  s, -s), (-s,  s, -s),
             (-s, -s,  s), ( s, -s,  s), ( s,  s,  s), (-s,  s,  s)]
    faces = [(0, 1, 2, 3), (4, 7, 6, 5),
             (0, 4, 5, 1), (1, 5, 6, 2),
             (2, 6, 7, 3), (3, 7, 4, 0)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.materials.append(_ensure_material(role))
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.hide_render = True
    coll.objects.link(obj)
    return obj


def _find_view_3d(context):
    """Return the active 3D viewport's region_3d, or None."""
    rd = getattr(context, "region_data", None)
    if rd is not None and hasattr(rd, "view_rotation"):
        return rd
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                rd = getattr(space, "region_3d", None)
                if rd is not None:
                    return rd
    return None


def _pick(coords, normals, view_dir_np, p_arr):
    """Pick the visible vertex closest to the line through p_arr along
    view_dir_np. Returns (vid, in_plane_distance, three_d_distance)."""
    diff = coords - p_arr
    cross = np.cross(diff, view_dir_np)
    d2 = np.einsum("ij,ij->i", cross, cross)
    dots = normals @ view_dir_np
    visible = dots < -EPSILON
    if not visible.any():
        visible = np.ones(len(coords), dtype=bool)
    d2[~visible] = np.inf
    vid = int(np.argmin(d2))
    inplane = float(np.sqrt(d2[vid])) if np.isfinite(d2[vid]) else float("inf")
    three_d = float(np.linalg.norm(coords[vid] - p_arr))
    return vid, inplane, three_d


class FLAMEVIEWER_OT_DebugPickFromView(bpy.types.Operator):
    bl_idname = "flameviewer.debug_pick_from_view"
    bl_label = "Pick from View"
    bl_description = ("Project the chosen bone's head and tail from the "
                      "current 3D viewport view direction onto the target "
                      "mesh; drop magenta (head) and cyan (tail) markers "
                      "at the picked verts. Reports per-endpoint in-plane "
                      "distance (image-space) and 3D distance — large 3D "
                      "distance ⇒ bone is nowhere near the mesh.")

    @classmethod
    def poll(cls, context):
        scene = context.scene
        arm = getattr(scene, "flame_viewer_armature", None)
        if arm is None or arm.type != "ARMATURE":
            return False
        bone_id = getattr(scene, "flame_viewer_debug_bone", "")
        if not bone_id or bone_id == "__none__":
            return False
        return get_target_mesh(scene) is not None

    def execute(self, context):
        scene = context.scene
        arm_obj = scene.flame_viewer_armature
        mesh_obj = get_target_mesh(scene)
        bone_name = scene.flame_viewer_debug_bone
        bone = arm_obj.data.bones.get(bone_name)
        if bone is None:
            self.report({"ERROR"}, f"Bone '{bone_name}' not in armature.")
            return {"CANCELLED"}

        rd = _find_view_3d(context)
        if rd is None:
            self.report({"ERROR"}, "Could not find a 3D viewport.")
            return {"CANCELLED"}

        # World-space view direction → mesh-local direction.
        look_world = rd.view_rotation @ Vector((0.0, 0.0, -1.0))
        look_world.normalize()
        mesh_inv = mesh_obj.matrix_world.inverted()
        look_mesh = mesh_inv.to_3x3() @ look_world
        look_mesh.normalize()
        view_dir_np = np.array([look_mesh.x, look_mesh.y, look_mesh.z],
                               dtype=np.float64)

        # Bone endpoints: armature-local → world → mesh-local.
        head_mesh = mesh_inv @ (arm_obj.matrix_world @ Vector(bone.head_local))
        tail_mesh = mesh_inv @ (arm_obj.matrix_world @ Vector(bone.tail_local))
        head_arr = np.array([head_mesh.x, head_mesh.y, head_mesh.z],
                            dtype=np.float64)
        tail_arr = np.array([tail_mesh.x, tail_mesh.y, tail_mesh.z],
                            dtype=np.float64)

        # Mesh verts + normals in mesh-local.
        mesh = mesh_obj.data
        n = len(mesh.vertices)
        coords = np.empty(n * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", coords)
        coords = coords.reshape(n, 3)
        normals = np.empty(n * 3, dtype=np.float64)
        mesh.vertices.foreach_get("normal", normals)
        normals = normals.reshape(n, 3)

        head_vid, head_inplane, head_3d = _pick(coords, normals,
                                                view_dir_np, head_arr)
        tail_vid, tail_inplane, tail_3d = _pick(coords, normals,
                                                view_dir_np, tail_arr)

        _clear()
        coll = _ensure_collection()
        head_loc = mesh_obj.matrix_world @ Vector(coords[head_vid].tolist())
        tail_loc = mesh_obj.matrix_world @ Vector(coords[tail_vid].tolist())
        _make_marker(f"{DEBUG_PREFIX}{bone_name}_head", "head", head_loc, coll)
        _make_marker(f"{DEBUG_PREFIX}{bone_name}_tail", "tail", tail_loc, coll)

        self.report(
            {"INFO"},
            f"{bone_name}: head v{head_vid} "
            f"(in-plane {head_inplane * 1000:.1f} mm, "
            f"3D {head_3d * 1000:.1f} mm); "
            f"tail v{tail_vid} "
            f"(in-plane {tail_inplane * 1000:.1f} mm, "
            f"3D {tail_3d * 1000:.1f} mm)")
        return {"FINISHED"}


class FLAMEVIEWER_OT_ClearDebugPicks(bpy.types.Operator):
    bl_idname = "flameviewer.clear_debug_picks"
    bl_label = "Clear Picks"
    bl_description = "Remove the debug-pick markers from the scene"

    def execute(self, context):
        _clear()
        return {"FINISHED"}


def register():
    bpy.utils.register_class(FLAMEVIEWER_OT_DebugPickFromView)
    bpy.utils.register_class(FLAMEVIEWER_OT_ClearDebugPicks)
    bpy.types.Scene.flame_viewer_debug_bone = bpy.props.EnumProperty(
        name="Bone",
        items=_bone_enum_items,
    )


def unregister():
    del bpy.types.Scene.flame_viewer_debug_bone
    bpy.utils.unregister_class(FLAMEVIEWER_OT_ClearDebugPicks)
    bpy.utils.unregister_class(FLAMEVIEWER_OT_DebugPickFromView)
