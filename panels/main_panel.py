"""Main sidebar panel for FLAME Viewer."""

import os

import bpy
from ..operators.update_mesh import _on_slider_change


def _armature_poll(self, obj):
    return obj.type == "ARMATURE"


def _mesh_poll(self, obj):
    return obj.type == "MESH"


class FLAMEVIEWER_PT_MainPanel(bpy.types.Panel):
    bl_label = "FLAME Viewer"
    bl_idname = "FLAMEVIEWER_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FLAME Viewer"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # ── Model ──
        box = layout.box()
        box.label(text="FLAME Model", icon="FILE")
        box.operator("flameviewer.load_model", text="Load PKL", icon="FILE_FOLDER")

        if scene.flame_viewer_loaded:
            box.label(text=f"  {os.path.basename(scene.flame_viewer_model_path)}")
            box.label(text=f"  {scene.flame_viewer_vert_count} verts, "
                           f"{scene.flame_viewer_face_count} faces")

        # ── Randomize ──
        box = layout.box()
        box.label(text="Randomize", icon="MOD_NOISE")
        row = box.row(align=True)
        row.operator("flameviewer.randomize", text="Randomize",
                     icon="MOD_NOISE")
        row.operator("flameviewer.reset", text="Reset", icon="LOOP_BACK")
        row = box.row(align=True)
        row.prop(scene, "flame_viewer_randomize_shape_1", toggle=True)
        row.prop(scene, "flame_viewer_randomize_shape_2", toggle=True)
        row.prop(scene, "flame_viewer_randomize_expression", toggle=True)
        row = box.row(align=True)
        row.prop(scene, "flame_viewer_randomize_global_rotation", toggle=True)
        row.prop(scene, "flame_viewer_randomize_jaw_pose", toggle=True)
        row.prop(scene, "flame_viewer_randomize_eye_pose", toggle=True)

        if not scene.flame_viewer_loaded:
            layout.label(text="Load a PKL file first.", icon="ERROR")
            return

        # ── Shape Parameters ──
        box = layout.box()
        box.label(text="Shape Parameters", icon="MESH_ICOSPHERE")
        for i in range(10):
            box.prop(scene, f'flame_viewer_shape_params[{i}]',
                     text=f"Shape {i + 1:03d}", slider=True)

        # ── Expression Parameters ──
        box = layout.box()
        box.label(text="Expression Parameters", icon="FACE_MAPS")
        for i in range(10):
            box.prop(scene, f'flame_viewer_expr_params[{i}]',
                     text=f"Expr {i + 1:03d}", slider=True)

        # ── Pose ──
        box = layout.box()
        box.label(text="Pose", icon="ARMATURE_DATA")

        for label, prop in (("Global Rotation:", "flame_viewer_global"),
                            ("Neck Rotation:",   "flame_viewer_neck"),
                            ("Jaw Rotation:",    "flame_viewer_jaw"),
                            ("Left Eye:",        "flame_viewer_eye_l"),
                            ("Right Eye:",       "flame_viewer_eye_r")):
            box.label(text=label)
            row = box.row(align=True)
            for axis in range(3):
                row.prop(scene, f"{prop}[{axis}]",
                         text="XYZ"[axis], slider=True)

        # ── Bone Tracking ──
        box = layout.box()
        box.label(text="Bone Tracking", icon="ARMATURE_DATA")
        box.prop(scene, "flame_viewer_target_mesh", text="Mesh")
        box.prop(scene, "flame_viewer_armature", text="Armature")
        box.operator("flameviewer.bake_anchors", text="Bake Anchors")
        if scene.flame_viewer_anchor_status:
            box.label(text=scene.flame_viewer_anchor_status, icon="INFO")
        row = box.row(align=True)
        row.operator("flameviewer.show_anchor_markers", text="Show Markers")
        row.operator("flameviewer.clear_anchor_markers", text="Clear Markers")

        box.separator()
        box.label(text="Debug Pick (current view)")
        box.prop(scene, "flame_viewer_debug_bone", text="Bone")
        row = box.row(align=True)
        row.operator("flameviewer.debug_pick_from_view", text="Pick from View")
        row.operator("flameviewer.clear_debug_picks", text="Clear Picks")


def register():
    bpy.utils.register_class(FLAMEVIEWER_PT_MainPanel)

    bpy.types.Scene.flame_viewer_loaded = bpy.props.BoolProperty(
        name="Loaded", default=False)
    bpy.types.Scene.flame_viewer_model_path = bpy.props.StringProperty(
        name="Model Path", default="")
    bpy.types.Scene.flame_viewer_vert_count = bpy.props.IntProperty(
        name="Vertices", default=0)
    bpy.types.Scene.flame_viewer_face_count = bpy.props.IntProperty(
        name="Faces", default=0)
    bpy.types.Scene.flame_viewer_randomize_shape_1 = bpy.props.BoolProperty(
        name="Shape 1-100",
        description="Randomize FLAME shape coefficients 0-99 (primary identity)",
        default=True,
    )
    bpy.types.Scene.flame_viewer_randomize_shape_2 = bpy.props.BoolProperty(
        name="Shape 100-300",
        description=("Randomize FLAME shape coefficients 100-299 "
                     "(secondary identity)"),
        default=True,
    )
    bpy.types.Scene.flame_viewer_randomize_expression = bpy.props.BoolProperty(
        name="Expression",
        description="Randomize FLAME expression coefficients 0-99",
        default=True,
    )
    bpy.types.Scene.flame_viewer_randomize_global_rotation = bpy.props.BoolProperty(
        name="Head Rot",
        description="Randomize global head rotation (sigma 0.2, clipped to ±0.5 rad)",
        default=False,
    )
    bpy.types.Scene.flame_viewer_randomize_jaw_pose = bpy.props.BoolProperty(
        name="Jaw",
        description="Randomize jaw opening (sigma 0.15, jaw axis only opens)",
        default=False,
    )
    bpy.types.Scene.flame_viewer_randomize_eye_pose = bpy.props.BoolProperty(
        name="Eyes",
        description="Randomize eye rotation per eye (sigma 0.1)",
        default=False,
    )

    bpy.types.Scene.flame_viewer_shape_params = bpy.props.FloatVectorProperty(
        name="Shape", size=10, default=[0.0] * 10,
        min=-3.0, max=3.0, subtype="NONE",
        update=_on_slider_change)

    bpy.types.Scene.flame_viewer_expr_params = bpy.props.FloatVectorProperty(
        name="Expression", size=10, default=[0.0] * 10,
        min=-3.0, max=3.0, subtype="NONE",
        update=_on_slider_change)

    bpy.types.Scene.flame_viewer_jaw = bpy.props.FloatVectorProperty(
        name="Jaw", size=3, default=[0.0] * 3,
        min=-1.0, max=1.0, subtype="EULER",
        update=_on_slider_change)

    bpy.types.Scene.flame_viewer_neck = bpy.props.FloatVectorProperty(
        name="Neck", size=3, default=[0.0] * 3,
        min=-1.0, max=1.0, subtype="EULER",
        update=_on_slider_change)

    bpy.types.Scene.flame_viewer_global = bpy.props.FloatVectorProperty(
        name="Global", size=3, default=[0.0] * 3,
        min=-1.0, max=1.0, subtype="EULER",
        update=_on_slider_change)

    bpy.types.Scene.flame_viewer_eye_l = bpy.props.FloatVectorProperty(
        name="Eye L", size=3, default=[0.0] * 3,
        min=-1.0, max=1.0, subtype="EULER",
        update=_on_slider_change)

    bpy.types.Scene.flame_viewer_eye_r = bpy.props.FloatVectorProperty(
        name="Eye R", size=3, default=[0.0] * 3,
        min=-1.0, max=1.0, subtype="EULER",
        update=_on_slider_change)

    bpy.types.Scene.flame_viewer_armature = bpy.props.PointerProperty(
        name="Armature",
        type=bpy.types.Object,
        poll=_armature_poll)
    bpy.types.Scene.flame_viewer_target_mesh = bpy.props.PointerProperty(
        name="Target Mesh",
        description=("Mesh to project bones onto (bake) and read coords "
                     "from each frame (runtime). Falls back to the auto-"
                     "created FLAME_Viewer mesh if blank. Switch to a "
                     "different mesh if your armature was authored "
                     "against an appended head, then bake; runtime "
                     "tracking only follows shape sliders when this is "
                     "FLAME_Viewer (the only mesh the addon deforms)."),
        type=bpy.types.Object,
        poll=_mesh_poll)
    bpy.types.Scene.flame_viewer_anchor_status = bpy.props.StringProperty(
        name="Anchor Status", default="")


def unregister():
    bpy.utils.unregister_class(FLAMEVIEWER_PT_MainPanel)

    del bpy.types.Scene.flame_viewer_anchor_status
    del bpy.types.Scene.flame_viewer_target_mesh
    del bpy.types.Scene.flame_viewer_armature
    del bpy.types.Scene.flame_viewer_eye_r
    del bpy.types.Scene.flame_viewer_eye_l
    del bpy.types.Scene.flame_viewer_global
    del bpy.types.Scene.flame_viewer_neck
    del bpy.types.Scene.flame_viewer_jaw
    del bpy.types.Scene.flame_viewer_expr_params
    del bpy.types.Scene.flame_viewer_shape_params
    del bpy.types.Scene.flame_viewer_randomize_eye_pose
    del bpy.types.Scene.flame_viewer_randomize_jaw_pose
    del bpy.types.Scene.flame_viewer_randomize_global_rotation
    del bpy.types.Scene.flame_viewer_randomize_expression
    del bpy.types.Scene.flame_viewer_randomize_shape_2
    del bpy.types.Scene.flame_viewer_randomize_shape_1
    del bpy.types.Scene.flame_viewer_face_count
    del bpy.types.Scene.flame_viewer_vert_count
    del bpy.types.Scene.flame_viewer_model_path
    del bpy.types.Scene.flame_viewer_loaded
