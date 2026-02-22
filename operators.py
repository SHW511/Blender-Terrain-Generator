"""
Operators for D&D Tile Forge.
Each operator corresponds to a major action in the plugin workflow.
"""

import os
import bmesh
import bpy
from bpy.types import Operator

from . import mesh_gen
from .mesh_gen import mm, m


# ---------------------------------------------------------------------------
# Preview / Generate full terrain
# ---------------------------------------------------------------------------

class TILEFORGE_OT_GenerateTerrain(Operator):
    """Generate the full terrain map as a single preview mesh"""
    bl_idname = "tileforge.generate_terrain"
    bl_label = "Generate Terrain"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        tf = context.scene.tile_forge
        tile = tf.tile
        outdoor = tf.outdoor
        print_scale = tile.print_scale

        # Clean up previous preview
        _remove_objects_by_prefix("TF_Preview")

        total_x = tile.map_width
        total_y = tile.map_depth

        obj, mesh = mesh_gen.create_base_grid(
            "TF_Preview_Terrain",
            total_x,
            total_y,
            outdoor.subdivisions,
            print_scale,
        )

        # Link to scene
        context.collection.objects.link(obj)
        context.view_layer.objects.active = obj
        obj.select_set(True)

        # Apply terrain
        is_procedural = outdoor.terrain_type != "CUSTOM"
        if not is_procedural and outdoor.heightmap_image:
            mesh_gen.apply_heightmap(obj, outdoor.heightmap_image, outdoor, tile)
        else:
            mesh_gen.apply_procedural_noise(obj, outdoor, tile)

        # Noise layers (procedural only)
        if is_procedural and outdoor.enable_noise_layers:
            mesh_gen.apply_noise_layers(obj, outdoor, tile)

        # Erosion (operates on height grid for performance)
        if outdoor.enable_hydraulic_erosion or outdoor.enable_thermal_erosion:
            mesh_data = obj.data
            bm = bmesh.new()
            bm.from_mesh(mesh_data)

            grid_size = outdoor.subdivisions + 1
            cell_x = m(tile.map_width, print_scale) / max(outdoor.subdivisions, 1)
            cell_y = m(tile.map_depth, print_scale) / max(outdoor.subdivisions, 1)
            cell_size = (cell_x + cell_y) * 0.5
            grid, index_map = mesh_gen.mesh_to_height_grid(bm, grid_size)

            if outdoor.enable_hydraulic_erosion:
                mesh_gen.apply_hydraulic_erosion(grid, grid_size, outdoor, cell_size)

            if outdoor.enable_thermal_erosion:
                mesh_gen.apply_thermal_erosion(grid, grid_size, outdoor, cell_size)

            mesh_gen.height_grid_to_mesh(bm, grid, grid_size, index_map)
            bm.to_mesh(mesh_data)
            bm.free()
            mesh_data.update()

        # Slope clamping (final safety pass — after erosion)
        if outdoor.enable_slope_clamp:
            mesh_gen.clamp_slopes(obj, outdoor.max_slope_angle)

        # River channel (after erosion/clamping — intentional carving)
        if outdoor.add_river and outdoor.river_curve is None:
            self.report({'WARNING'}, "River enabled but no curve assigned — skipping")
        mesh_gen.apply_river_channel(obj, outdoor, tile)

        # Path / road (after erosion — flattens to average height)
        if outdoor.add_path and outdoor.path_curve is None:
            self.report({'WARNING'}, "Path enabled but no curve assigned — skipping")
        mesh_gen.apply_path(obj, outdoor, tile)

        # Ground texture (last detail pass — micro-displacement)
        if outdoor.floor_texture != 'NONE':
            mesh_gen.apply_ground_texture(
                obj, outdoor.floor_texture, outdoor.texture_strength, tile
            )

        # Add solid base
        mesh_gen.add_solid_base(obj, tile.base_height)

        # Assign terrain material
        mesh_gen.assign_terrain_material(obj)

        self.report({'INFO'}, f"Generated terrain: {total_x:.1f} x {total_y:.1f} m")
        return {'FINISHED'}


class TILEFORGE_OT_GenerateDungeon(Operator):
    """Generate dungeon tiles from a floor plan image"""
    bl_idname = "tileforge.generate_dungeon"
    bl_label = "Generate Dungeon"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        tf = context.scene.tile_forge
        tile = tf.tile
        dungeon = tf.dungeon
        print_scale = tile.print_scale

        if not dungeon.floorplan_image:
            self.report({'ERROR'}, "No floor plan image specified")
            return {'CANCELLED'}

        # Clean up previous
        _remove_objects_by_prefix("TF_Preview")

        total_x = tile.map_width
        total_y = tile.map_depth

        obj, mesh = mesh_gen.create_base_grid(
            "TF_Preview_Dungeon",
            total_x,
            total_y,
            max(64, int(total_x / tile.tile_size_x) * 32),
            print_scale,
        )

        context.collection.objects.link(obj)
        context.view_layer.objects.active = obj
        obj.select_set(True)

        # Generate dungeon from floor plan
        # For preview, generate the whole thing at once
        img = bpy.data.images.load(dungeon.floorplan_image, check_existing=True)
        pixels = list(img.pixels)
        img_w, img_h = img.size

        import bmesh
        bm = bmesh.new()
        bm.from_mesh(mesh)

        half_x = m(total_x, print_scale) / 2.0
        half_y = m(total_y, print_scale) / 2.0
        wall_h = m(dungeon.wall_height, print_scale)

        for v in bm.verts:
            u = (v.co.x + half_x) / (2.0 * half_x) if half_x > 0 else 0.5
            v_coord = (v.co.y + half_y) / (2.0 * half_y) if half_y > 0 else 0.5
            u = max(0.0, min(1.0, u))
            v_coord = max(0.0, min(1.0, v_coord))

            px = int(u * (img_w - 1))
            py = int(v_coord * (img_h - 1))
            idx = (py * img_w + px) * 4

            brightness = pixels[idx] if idx < len(pixels) else 1.0
            v.co.z = wall_h if brightness < 0.5 else 0.0

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        # Floor texture (only on floor verts, not walls)
        if dungeon.floor_texture != 'NONE':
            mesh_gen.apply_ground_texture(
                obj, dungeon.floor_texture, dungeon.texture_strength, tile,
                floor_only_z=mm(1.0),
            )

        mesh_gen.add_solid_base(obj, tile.base_height)

        self.report({'INFO'}, f"Generated dungeon: {total_x:.1f} x {total_y:.1f} m")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Slice into tiles
# ---------------------------------------------------------------------------

class TILEFORGE_OT_SliceTiles(Operator):
    """Slice the preview terrain into individual modular tiles"""
    bl_idname = "tileforge.slice_tiles"
    bl_label = "Slice into Tiles"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        tf = context.scene.tile_forge
        tile = tf.tile
        print_scale = tile.print_scale

        # Find the preview object
        preview_obj = None
        for obj in context.scene.objects:
            if obj.name.startswith("TF_Preview"):
                preview_obj = obj
                break

        if not preview_obj:
            self.report({'ERROR'}, "No terrain preview found. Generate terrain first.")
            return {'CANCELLED'}

        # Clean up old tiles
        _remove_objects_by_prefix("tile_r")

        # Auto-calculate tile counts from map size and tile size
        import math
        cols = max(1, math.floor(tile.map_width / tile.tile_size_x))
        rows = max(1, math.floor(tile.map_depth / tile.tile_size_y))
        tile.map_tiles_x = cols
        tile.map_tiles_y = rows

        tile_count = 0
        for row in range(rows):
            for col in range(cols):
                tile_obj = mesh_gen.slice_terrain_to_tile(
                    preview_obj, tile, col, row
                )

                # Add connectors
                mesh_gen.add_connectors(tile_obj, tile, col, row)

                # Engrave grid lines
                mesh_gen.engrave_grid_lines(tile_obj, tile)

                tile_count += 1

        # Arrange tiles in a grid for viewport display
        spacing = m(tile.tile_size_x, print_scale) * 1.2
        for obj in context.scene.objects:
            if obj.name.startswith("tile_r"):
                parts = obj.name.split("_")
                try:
                    r = int(parts[1][1:])
                    c = int(parts[2][1:])
                    obj.location.x = c * spacing
                    obj.location.y = r * spacing
                except (IndexError, ValueError):
                    pass

        # Optionally hide the preview
        preview_obj.hide_set(True)

        self.report({'INFO'}, f"Sliced into {tile_count} tiles")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Randomize seed
# ---------------------------------------------------------------------------

class TILEFORGE_OT_RandomizeSeed(Operator):
    """Randomize the noise seed for procedural generation"""
    bl_idname = "tileforge.randomize_seed"
    bl_label = "Randomize Seed"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import random
        context.scene.tile_forge.outdoor.noise_seed = random.randint(0, 999999)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Export STL
# ---------------------------------------------------------------------------

class TILEFORGE_OT_ExportTiles(Operator):
    """Export all tiles as individual STL files"""
    bl_idname = "tileforge.export_tiles"
    bl_label = "Export Tiles (STL)"
    bl_options = {'REGISTER'}

    def execute(self, context):
        tf = context.scene.tile_forge
        export = tf.export

        # Resolve export path
        export_path = bpy.path.abspath(export.export_path)
        if not export_path:
            self.report({'ERROR'}, "No export path specified")
            return {'CANCELLED'}

        os.makedirs(export_path, exist_ok=True)

        # Find all tile objects
        tile_objects = [
            obj for obj in context.scene.objects
            if obj.name.startswith("tile_r") and obj.type == 'MESH'
        ]

        if not tile_objects:
            self.report({'ERROR'}, "No tiles found. Slice terrain first.")
            return {'CANCELLED'}

        exported = 0
        warnings = []

        for obj in tile_objects:
            # Manifold check
            if export.check_manifold:
                is_ok, bad_edges, bad_verts = mesh_gen.check_manifold(obj)
                if not is_ok:
                    warnings.append(
                        f"{obj.name}: {bad_edges} non-manifold edges, "
                        f"{bad_verts} non-manifold verts"
                    )

            # Select only this object
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj

            filename = f"{export.filename_prefix}_{obj.name}.stl"
            filepath = os.path.join(export_path, filename)

            bpy.ops.export_mesh.stl(
                filepath=filepath,
                use_selection=True,
                use_mesh_modifiers=True,
                global_scale=1000.0,  # Blender meters -> mm for slicers
                ascii=False,
            )
            exported += 1

        # Report results
        if warnings:
            for w in warnings:
                self.report({'WARNING'}, w)

        self.report({'INFO'}, f"Exported {exported} tiles to {export_path}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Export single tile
# ---------------------------------------------------------------------------

class TILEFORGE_OT_ExportSingleTile(Operator):
    """Export the selected tile as STL"""
    bl_idname = "tileforge.export_single_tile"
    bl_label = "Export Selected Tile"
    bl_options = {'REGISTER'}

    def execute(self, context):
        tf = context.scene.tile_forge
        export = tf.export

        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object to export")
            return {'CANCELLED'}

        export_path = bpy.path.abspath(export.export_path)
        os.makedirs(export_path, exist_ok=True)

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj

        filename = f"{export.filename_prefix}_{obj.name}.stl"
        filepath = os.path.join(export_path, filename)

        bpy.ops.export_mesh.stl(
            filepath=filepath,
            use_selection=True,
            use_mesh_modifiers=True,
            global_scale=1000.0,
            ascii=False,
        )

        self.report({'INFO'}, f"Exported: {filename}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Noise layer management
# ---------------------------------------------------------------------------

class TILEFORGE_OT_AddNoiseLayer(Operator):
    """Add a noise layer to the terrain"""
    bl_idname = "tileforge.add_noise_layer"
    bl_label = "Add Noise Layer"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        outdoor = context.scene.tile_forge.outdoor
        if len(outdoor.noise_layers) >= 3:
            self.report({'WARNING'}, "Maximum 3 noise layers allowed")
            return {'CANCELLED'}
        layer = outdoor.noise_layers.add()
        layer.layer_name = f"Layer {len(outdoor.noise_layers)}"
        layer.seed_offset = len(outdoor.noise_layers) * 1000
        outdoor.active_noise_layer_index = len(outdoor.noise_layers) - 1
        return {'FINISHED'}


class TILEFORGE_OT_RemoveNoiseLayer(Operator):
    """Remove the active noise layer"""
    bl_idname = "tileforge.remove_noise_layer"
    bl_label = "Remove Noise Layer"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        outdoor = context.scene.tile_forge.outdoor
        idx = outdoor.active_noise_layer_index
        if idx < 0 or idx >= len(outdoor.noise_layers):
            self.report({'WARNING'}, "No noise layer selected")
            return {'CANCELLED'}
        outdoor.noise_layers.remove(idx)
        outdoor.active_noise_layer_index = min(idx, len(outdoor.noise_layers) - 1)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Cleanup utility
# ---------------------------------------------------------------------------

class TILEFORGE_OT_CleanupAll(Operator):
    """Remove all Tile Forge generated objects from the scene"""
    bl_idname = "tileforge.cleanup_all"
    bl_label = "Clean Up All Tiles"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        count = 0
        count += _remove_objects_by_prefix("TF_Preview")
        count += _remove_objects_by_prefix("tile_r")
        self.report({'INFO'}, f"Removed {count} objects")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remove_objects_by_prefix(prefix):
    """Remove all objects whose name starts with prefix. Returns count removed."""
    to_remove = [
        obj for obj in bpy.data.objects if obj.name.startswith(prefix)
    ]
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)
    return len(to_remove)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = (
    TILEFORGE_OT_GenerateTerrain,
    TILEFORGE_OT_GenerateDungeon,
    TILEFORGE_OT_SliceTiles,
    TILEFORGE_OT_RandomizeSeed,
    TILEFORGE_OT_ExportTiles,
    TILEFORGE_OT_ExportSingleTile,
    TILEFORGE_OT_AddNoiseLayer,
    TILEFORGE_OT_RemoveNoiseLayer,
    TILEFORGE_OT_CleanupAll,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
