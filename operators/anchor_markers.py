"""In-viewport debug markers for triplanar anchors.

For each baked endpoint we drop 4 small colored cubes:
    front-view pick (red), right-view pick (green), top-view pick (blue),
    reconstructed endpoint (yellow).

Each marker carries its own vertex-id (or full triplanar id triple, for
the recon marker) as a custom property, so the runtime update loop is
decoupled from the JSON on the armature — markers can be sanity-checked
even after a re-bake without re-reading the anchor dict.
"""

import json
import bpy
import numpy as np
from mathutils import Vector

from .update_bones import get_target_mesh

MESH_NAME = "FLAME_Viewer"
MARKER_COLL_NAME = "FLAMEViewer_Anchors"
MARKER_PREFIX = "FLAMEAnchor_"
MARKER_SIZE = 0.003  # half-edge of the cube; ~3 mm on a 0.2 m head

VID_KEY = "flameviewer_marker_vid"
RECON_KEY = "flameviewer_marker_recon"

ROLE_COLORS = {
    "front": (1.0, 0.15, 0.15, 1.0),
    "right": (0.15, 1.0, 0.20, 1.0),
    "top":   (0.30, 0.50, 1.0, 1.0),
    "recon": (1.0, 0.95, 0.15, 1.0),
}


def _get_or_create_material(role):
    name = f"FLAMEAnchor_Mat_{role}"
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name=name)
    mat.diffuse_color = ROLE_COLORS[role]
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = ROLE_COLORS[role]
    em.inputs["Strength"].default_value = 3.0
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def _get_or_create_collection():
    coll = bpy.data.collections.get(MARKER_COLL_NAME)
    if coll is None:
        coll = bpy.data.collections.new(MARKER_COLL_NAME)
        bpy.context.scene.collection.children.link(coll)
    return coll


def _make_cube_mesh(name):
    s = MARKER_SIZE
    verts = [(-s, -s, -s), ( s, -s, -s), ( s,  s, -s), (-s,  s, -s),
             (-s, -s,  s), ( s, -s,  s), ( s,  s,  s), (-s,  s,  s)]
    faces = [(0, 1, 2, 3), (4, 7, 6, 5),
             (0, 4, 5, 1), (1, 5, 6, 2),
             (2, 6, 7, 3), (3, 7, 4, 0)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def _make_marker(name, role, location, coll):
    mesh = _make_cube_mesh(name)
    mesh.materials.append(_get_or_create_material(role))
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.hide_render = True
    coll.objects.link(obj)
    return obj


def clear_markers():
    coll = bpy.data.collections.get(MARKER_COLL_NAME)
    if coll is None:
        return
    for obj in list(coll.objects):
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    bpy.data.collections.remove(coll)


def create_markers(context):
    scene = context.scene
    arm_obj = getattr(scene, "flame_viewer_armature", None)
    if arm_obj is None or "flame_viewer_anchors" not in arm_obj.keys():
        return 0

    mesh_obj = get_target_mesh(context.scene)
    if mesh_obj is None:
        return 0

    try:
        anchors = json.loads(arm_obj["flame_viewer_anchors"])
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0

    mesh = mesh_obj.data
    n = len(mesh.vertices)
    coords = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape(n, 3)

    clear_markers()
    coll = _get_or_create_collection()
    mesh_world = mesh_obj.matrix_world

    count = 0
    for bone_name, sides in anchors.items():
        for side_name in ("head", "tail"):
            ids = sides[side_name]
            for view in ("front", "right", "top"):
                vid = int(ids[view])
                if vid < 0 or vid >= n:
                    continue
                loc = mesh_world @ Vector(coords[vid].tolist())
                m = _make_marker(
                    f"{MARKER_PREFIX}{bone_name}_{side_name}_{view}",
                    view, loc, coll,
                )
                m[VID_KEY] = vid
                count += 1
            recon_local = (
                float(coords[ids["top"]][0]),
                float(coords[ids["right"]][1]),
                float(coords[ids["front"]][2]),
            )
            recon_loc = mesh_world @ Vector(recon_local)
            m = _make_marker(
                f"{MARKER_PREFIX}{bone_name}_{side_name}_recon",
                "recon", recon_loc, coll,
            )
            m[VID_KEY] = -1
            m[RECON_KEY] = json.dumps({k: int(ids[k])
                                       for k in ("front", "right", "top")})
            count += 1
    return count


def update_markers(context):
    coll = bpy.data.collections.get(MARKER_COLL_NAME)
    if coll is None or not coll.objects:
        return
    mesh_obj = get_target_mesh(context.scene)
    if mesh_obj is None:
        return

    mesh = mesh_obj.data
    n = len(mesh.vertices)
    coords = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape(n, 3)
    mesh_world = mesh_obj.matrix_world

    for marker in coll.objects:
        vid = marker.get(VID_KEY)
        if vid is None:
            continue
        vid = int(vid)
        if vid >= 0:
            if vid < n:
                marker.location = mesh_world @ Vector(coords[vid].tolist())
            continue
        recon_raw = marker.get(RECON_KEY)
        if recon_raw is None:
            continue
        try:
            ids = json.loads(recon_raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if max(ids["top"], ids["right"], ids["front"]) >= n:
            continue
        recon_local = (
            float(coords[ids["top"]][0]),
            float(coords[ids["right"]][1]),
            float(coords[ids["front"]][2]),
        )
        marker.location = mesh_world @ Vector(recon_local)


class FLAMEVIEWER_OT_ShowAnchorMarkers(bpy.types.Operator):
    bl_idname = "flameviewer.show_anchor_markers"
    bl_label = "Show Markers"
    bl_description = ("Drop colored markers at each triplanar anchor: "
                      "red = front-view pick, green = right-view pick, "
                      "blue = top-view pick, yellow = reconstructed endpoint")

    @classmethod
    def poll(cls, context):
        scene = context.scene
        arm = getattr(scene, "flame_viewer_armature", None)
        if arm is None or arm.type != "ARMATURE":
            return False
        return "flame_viewer_anchors" in arm.keys()

    def execute(self, context):
        n = create_markers(context)
        if n == 0:
            self.report({"WARNING"}, "No anchors found — bake first.")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Created {n} anchor markers.")
        return {"FINISHED"}


class FLAMEVIEWER_OT_ClearAnchorMarkers(bpy.types.Operator):
    bl_idname = "flameviewer.clear_anchor_markers"
    bl_label = "Clear Markers"
    bl_description = "Remove the anchor visualization markers from the scene"

    def execute(self, context):
        clear_markers()
        return {"FINISHED"}


def register():
    bpy.utils.register_class(FLAMEVIEWER_OT_ShowAnchorMarkers)
    bpy.utils.register_class(FLAMEVIEWER_OT_ClearAnchorMarkers)


def unregister():
    bpy.utils.unregister_class(FLAMEVIEWER_OT_ClearAnchorMarkers)
    bpy.utils.unregister_class(FLAMEVIEWER_OT_ShowAnchorMarkers)
