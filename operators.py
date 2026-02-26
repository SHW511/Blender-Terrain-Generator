"""
Operators for D&D Tile Forge.
Each operator corresponds to a major action in the plugin workflow.
"""

import os
import bmesh
import bpy
from bpy.types import Operator

from . import ai_api
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

        if outdoor.is_road_painting:
            self.report({'ERROR'}, "Exit road paint mode before generating")
            return {'CANCELLED'}

        wm = context.window_manager
        wm.progress_begin(0, 14)

        try:
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
            wm.progress_update(1)

            # Apply terrain
            if outdoor.terrain_type == "CUSTOM" and outdoor.heightmap_image:
                mesh_gen.apply_heightmap(obj, outdoor.heightmap_image, outdoor, tile)
            elif outdoor.terrain_type == "MAP_IMAGE" and outdoor.heightmap_image:
                mesh_gen.apply_color_map(obj, outdoor.heightmap_image, outdoor, tile)
                if outdoor.enable_terracing:
                    mesh_gen.apply_terracing_post(obj, outdoor, tile)
            else:
                mesh_gen.apply_procedural_noise(obj, outdoor, tile)
            wm.progress_update(2)

            # Noise layers (procedural modes only)
            is_image_mode = outdoor.terrain_type in ("CUSTOM", "MAP_IMAGE")
            if not is_image_mode and outdoor.enable_noise_layers:
                mesh_gen.apply_noise_layers(obj, outdoor, tile)
            wm.progress_update(3)

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
            wm.progress_update(4)

            # Slope clamping (final safety pass — after erosion)
            if outdoor.enable_slope_clamp:
                mesh_gen.clamp_slopes(obj, outdoor.max_slope_angle)
            wm.progress_update(5)

            # Cliff (large-scale terrain reshaping — before additive features)
            if outdoor.add_cliff and outdoor.cliff_curve is None:
                self.report({'WARNING'}, "Cliff enabled but no curve assigned — skipping")
            mesh_gen.apply_cliff(obj, outdoor, tile)
            wm.progress_update(6)

            # Ridge line (additive feature)
            if outdoor.add_ridge and outdoor.ridge_curve is None:
                self.report({'WARNING'}, "Ridge enabled but no curve assigned — skipping")
            mesh_gen.apply_ridge_line(obj, outdoor, tile)
            wm.progress_update(7)

            # River channel (after erosion/clamping — intentional carving)
            if outdoor.add_river and outdoor.river_curve is None:
                self.report({'WARNING'}, "River enabled but no curve assigned — skipping")
            mesh_gen.apply_river_channel(obj, outdoor, tile)
            wm.progress_update(8)

            # Path / road (after erosion — flattens to average height)
            if outdoor.add_path and outdoor.path_curve is None:
                self.report({'WARNING'}, "Path enabled but no curve assigned — skipping")
            mesh_gen.apply_path(obj, outdoor, tile)
            wm.progress_update(9)

            # Procedural road network (A* pathfinding between waypoints)
            if outdoor.add_road_network:
                for seg in outdoor.road_segments:
                    if seg.enabled and (seg.waypoint_start is None or seg.waypoint_end is None):
                        self.report({'WARNING'}, f"Road '{seg.segment_name}': missing waypoints — skipping")
                mesh_gen.apply_road_network(obj, outdoor, tile)
            wm.progress_update(10)

            # Painted road mask (flatten terrain along painted strokes)
            mesh_gen.apply_painted_road(obj, outdoor, tile)
            wm.progress_update(11)

            # Ground texture (last detail pass — micro-displacement)
            road_tex_kwargs = {}
            if (outdoor.add_painted_road
                    and outdoor.road_paint_texture != 'NONE'):
                mask_img = bpy.data.images.get("TF_RoadPaint_Mask")
                if mask_img:
                    road_tex_kwargs = {
                        'road_mask_pixels': list(mask_img.pixels),
                        'road_mask_w': mask_img.size[0],
                        'road_mask_h': mask_img.size[1],
                        'road_texture': outdoor.road_paint_texture,
                        'road_texture_strength': outdoor.road_paint_texture_strength,
                        'road_texture_scale': outdoor.road_paint_texture_scale,
                        'road_cobble_density': outdoor.road_paint_cobble_density,
                    }
            if outdoor.floor_texture != 'NONE' or road_tex_kwargs:
                mesh_gen.apply_ground_texture(
                    obj, outdoor.floor_texture, outdoor.texture_strength, tile,
                    **road_tex_kwargs,
                )
            wm.progress_update(12)

            # Add solid base
            mesh_gen.add_solid_base(obj, tile.base_height)
            wm.progress_update(13)

            # Smooth shading for higher quality terrain surface
            mesh_gen.apply_shade_smooth(obj)

            # Assign terrain material
            mesh_gen.assign_terrain_material(obj)
            wm.progress_update(14)

            self.report({'INFO'}, f"Generated terrain: {total_x:.1f} x {total_y:.1f} m")
            return {'FINISHED'}
        finally:
            wm.progress_end()


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

        wm = context.window_manager
        total_tiles = rows * cols
        wm.progress_begin(0, total_tiles)

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
                wm.progress_update(tile_count)

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

        wm.progress_end()
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

        wm = context.window_manager
        wm.progress_begin(0, len(tile_objects))

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

            bpy.ops.wm.stl_export(
                filepath=filepath,
                export_selected_objects=True,
                global_scale=1000.0,  # Blender meters -> mm for slicers
                ascii_format=False,
            )
            exported += 1
            wm.progress_update(exported)

        wm.progress_end()

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

        bpy.ops.wm.stl_export(
            filepath=filepath,
            export_selected_objects=True,
            global_scale=1000.0,
            ascii_format=False,
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
# Color zone management
# ---------------------------------------------------------------------------

class TILEFORGE_OT_AddColorZone(Operator):
    """Add a color zone to the color map terrain"""
    bl_idname = "tileforge.add_color_zone"
    bl_label = "Add Color Zone"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        outdoor = context.scene.tile_forge.outdoor
        if len(outdoor.color_zones) >= 8:
            self.report({'WARNING'}, "Maximum 8 color zones allowed")
            return {'CANCELLED'}

        # Auto-populate defaults on first add
        if len(outdoor.color_zones) == 0:
            defaults = [
                ("Water",    (0.55, 0.75, 0.80), 0.35, 0.0),
                ("Lowland",  (0.45, 0.60, 0.35), 0.30, 0.35),
                ("Highland", (0.50, 0.40, 0.30), 0.30, 0.80),
            ]
            for name, color, tol, height in defaults:
                zone = outdoor.color_zones.add()
                zone.zone_name = name
                zone.color = color
                zone.tolerance = tol
                zone.height = height
        else:
            zone = outdoor.color_zones.add()
            zone.zone_name = f"Zone {len(outdoor.color_zones)}"

        outdoor.active_color_zone_index = len(outdoor.color_zones) - 1
        return {'FINISHED'}


class TILEFORGE_OT_RemoveColorZone(Operator):
    """Remove the active color zone"""
    bl_idname = "tileforge.remove_color_zone"
    bl_label = "Remove Color Zone"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        outdoor = context.scene.tile_forge.outdoor
        idx = outdoor.active_color_zone_index
        if idx < 0 or idx >= len(outdoor.color_zones):
            self.report({'WARNING'}, "No color zone selected")
            return {'CANCELLED'}
        outdoor.color_zones.remove(idx)
        outdoor.active_color_zone_index = min(idx, len(outdoor.color_zones) - 1)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Heightmap painting
# ---------------------------------------------------------------------------

class TILEFORGE_OT_AddRoadSegment(Operator):
    """Add a road segment to the road network"""
    bl_idname = "tileforge.add_road_segment"
    bl_label = "Add Road Segment"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        outdoor = context.scene.tile_forge.outdoor
        if len(outdoor.road_segments) >= 8:
            self.report({'WARNING'}, "Maximum 8 road segments allowed")
            return {'CANCELLED'}
        seg = outdoor.road_segments.add()
        seg.segment_name = f"Road {len(outdoor.road_segments)}"
        outdoor.active_road_segment_index = len(outdoor.road_segments) - 1
        return {'FINISHED'}


class TILEFORGE_OT_RemoveRoadSegment(Operator):
    """Remove the active road segment"""
    bl_idname = "tileforge.remove_road_segment"
    bl_label = "Remove Road Segment"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        outdoor = context.scene.tile_forge.outdoor
        idx = outdoor.active_road_segment_index
        if idx < 0 or idx >= len(outdoor.road_segments):
            self.report({'WARNING'}, "No road segment selected")
            return {'CANCELLED'}
        outdoor.road_segments.remove(idx)
        outdoor.active_road_segment_index = min(idx, len(outdoor.road_segments) - 1)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Heightmap painting
# ---------------------------------------------------------------------------

class TILEFORGE_OT_SetupHeightmapPaint(Operator):
    """Set up a textured plane for painting elevation zones over a reference map"""
    bl_idname = "tileforge.setup_heightmap_paint"
    bl_label = "Start Painting"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        outdoor = context.scene.tile_forge.outdoor

        if outdoor.is_road_painting:
            self.report({'ERROR'}, "Exit road paint mode first")
            return {'CANCELLED'}

        ref_path = bpy.path.abspath(outdoor.paint_reference_image)
        if not ref_path or not os.path.isfile(ref_path):
            self.report({'ERROR'}, "Reference map image not found")
            return {'CANCELLED'}

        # Clean up any previous paint objects
        _remove_objects_by_prefix("TF_Paint_")

        # Load reference image
        ref_img = bpy.data.images.load(ref_path, check_existing=True)
        img_w, img_h = ref_img.size

        # Create or reuse paint image (preserves previous strokes on re-entry)
        res = outdoor.paint_resolution
        paint_img = bpy.data.images.get("TF_Paint_Heightmap")
        if paint_img and (paint_img.size[0] != res or paint_img.size[1] != res):
            bpy.data.images.remove(paint_img)
            paint_img = None
        if not paint_img:
            paint_img = bpy.data.images.new(
                "TF_Paint_Heightmap", width=res, height=res,
                alpha=True, float_buffer=False,
            )
            # Start fully transparent so the reference shows through
            paint_img.pixels[:] = [0.0] * (res * res * 4)

            # Load saved progress from disk if available
            progress_path = _paint_progress_path(outdoor)
            if progress_path and os.path.isfile(progress_path):
                saved = bpy.data.images.load(progress_path, check_existing=False)
                if saved.size[0] == res and saved.size[1] == res:
                    paint_img.pixels[:] = list(saved.pixels)
                bpy.data.images.remove(saved)
        paint_img.use_fake_user = True

        # Create plane matching image aspect ratio
        aspect = img_w / max(img_h, 1)
        size_y = 2.0
        size_x = size_y * aspect

        bm = bmesh.new()
        verts = [
            bm.verts.new((-size_x / 2, -size_y / 2, 0)),
            bm.verts.new(( size_x / 2, -size_y / 2, 0)),
            bm.verts.new(( size_x / 2,  size_y / 2, 0)),
            bm.verts.new((-size_x / 2,  size_y / 2, 0)),
        ]
        face = bm.faces.new(verts)

        # UV mapping
        uv_layer = bm.loops.layers.uv.new("UVMap")
        uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
        for loop, uv in zip(face.loops, uvs):
            loop[uv_layer].uv = uv

        mesh = bpy.data.meshes.new("TF_Paint_Plane_Mesh")
        bm.to_mesh(mesh)
        bm.free()

        plane_obj = bpy.data.objects.new("TF_Paint_Plane", mesh)
        context.collection.objects.link(plane_obj)

        # Build material with reference + paint overlay
        mat = bpy.data.materials.new("TF_Paint_Material")
        mat.use_nodes = True
        tree = mat.node_tree
        tree.nodes.clear()

        # Nodes
        n_ref = tree.nodes.new('ShaderNodeTexImage')
        n_ref.name = "TF_RefTex"
        n_ref.image = ref_img
        n_ref.location = (-600, 300)

        n_paint = tree.nodes.new('ShaderNodeTexImage')
        n_paint.name = "TF_PaintTex"
        n_paint.image = paint_img
        n_paint.location = (-600, -100)

        n_mult = tree.nodes.new('ShaderNodeMath')
        n_mult.name = "TF_OpacityMult"
        n_mult.operation = 'MULTIPLY'
        n_mult.inputs[1].default_value = outdoor.paint_overlay_opacity
        n_mult.location = (-100, -200)

        n_mix = tree.nodes.new('ShaderNodeMix')
        n_mix.name = "TF_MixColor"
        n_mix.data_type = 'RGBA'
        n_mix.location = (100, 200)

        n_bsdf = tree.nodes.new('ShaderNodeBsdfPrincipled')
        n_bsdf.location = (400, 200)

        n_out = tree.nodes.new('ShaderNodeOutputMaterial')
        n_out.location = (700, 200)

        # Links
        links = tree.links
        links.new(n_paint.outputs['Alpha'], n_mult.inputs[0])
        links.new(n_mult.outputs['Value'], n_mix.inputs['Factor'])
        links.new(n_ref.outputs['Color'], n_mix.inputs[6])   # A input
        links.new(n_paint.outputs['Color'], n_mix.inputs[7])  # B input
        links.new(n_mix.outputs[2], n_bsdf.inputs['Base Color'])  # Result
        links.new(n_bsdf.outputs['BSDF'], n_out.inputs['Surface'])

        # Set paint texture as active for texture paint mode
        mat.node_tree.nodes.active = n_paint

        plane_obj.data.materials.append(mat)

        # Select plane and enter texture paint mode
        bpy.ops.object.select_all(action='DESELECT')
        plane_obj.select_set(True)
        context.view_layer.objects.active = plane_obj
        bpy.ops.object.mode_set(mode='TEXTURE_PAINT')

        # Configure brush for mid-gray default
        ip = context.tool_settings.image_paint
        brush = ip.brush
        if brush:
            brush.color = (0.5, 0.5, 0.5)
            brush.blend = 'MIX'
            ups = ip.unified_paint_settings
            if ups.use_unified_color:
                ups.color = (0.5, 0.5, 0.5)

        # Switch viewport to Material Preview
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'MATERIAL'
                        break
                break

        outdoor.is_painting = True
        self.report({'INFO'}, "Paint elevation zones on the map — use height level buttons")
        return {'FINISHED'}


class TILEFORGE_OT_SetBrushHeight(Operator):
    """Set the paint brush to a specific elevation level"""
    bl_idname = "tileforge.set_brush_height"
    bl_label = "Set Brush Height"
    bl_options = {'REGISTER'}

    level: bpy.props.EnumProperty(
        name="Level",
        items=[
            ("SEA",    "Sea",    "Lowest elevation — black (0.0)"),
            ("LOW",    "Low",    "Low elevation (0.25)"),
            ("MID",    "Mid",    "Mid elevation (0.5)"),
            ("HIGH",   "High",   "High elevation (0.75)"),
            ("PEAK",   "Peak",   "Highest elevation — white (1.0)"),
            ("ERASER", "Eraser", "Erase painted areas to reveal reference"),
        ],
    )

    _gray_values = {"SEA": 0.0, "LOW": 0.25, "MID": 0.5, "HIGH": 0.75, "PEAK": 1.0}

    @classmethod
    def poll(cls, context):
        return context.scene.tile_forge.outdoor.is_painting

    def execute(self, context):
        # Ensure we're in texture paint mode
        if context.mode != 'PAINT_TEXTURE':
            obj = context.active_object
            if obj:
                bpy.ops.object.mode_set(mode='TEXTURE_PAINT')

        brush = context.tool_settings.image_paint.brush
        if not brush:
            self.report({'WARNING'}, "No active paint brush")
            return {'CANCELLED'}

        if self.level == "ERASER":
            brush.blend = 'ERASE_ALPHA'
        else:
            brush.blend = 'MIX'
            g = self._gray_values[self.level]
            color = (g, g, g)
            brush.color = color
            # Blender 5.0 moved unified_paint_settings into each Paint struct
            ups = context.tool_settings.image_paint.unified_paint_settings
            if ups.use_unified_color:
                ups.color = color

        return {'FINISHED'}


class TILEFORGE_OT_ApplyPaintedHeightmap(Operator):
    """Convert the painted overlay to a grayscale heightmap and set it as Custom Heightmap"""
    bl_idname = "tileforge.apply_painted_heightmap"
    bl_label = "Apply Heightmap"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.tile_forge.outdoor.is_painting

    def execute(self, context):
        outdoor = context.scene.tile_forge.outdoor

        try:
            # Exit paint mode
            if context.mode == 'PAINT_TEXTURE':
                bpy.ops.object.mode_set(mode='OBJECT')

            paint_img = bpy.data.images.get("TF_Paint_Heightmap")
            if not paint_img:
                self.report({'ERROR'}, "No painted heightmap found")
                return {'CANCELLED'}

            # Convert RGBA to grayscale PNG
            w, h = paint_img.size
            px = list(paint_img.pixels)  # flat RGBA
            gray_pixels = [0.0] * (w * h * 4)

            for i in range(w * h):
                r = px[i * 4]
                a = px[i * 4 + 3]
                # Painted areas use their RGB value; unpainted (alpha=0) default to mid
                g = r * a + 0.5 * (1.0 - a)
                gray_pixels[i * 4]     = g
                gray_pixels[i * 4 + 1] = g
                gray_pixels[i * 4 + 2] = g
                gray_pixels[i * 4 + 3] = 1.0

            # Create output image
            out_img = bpy.data.images.new(
                "TF_Paint_Output", width=w, height=h,
                alpha=False, float_buffer=False,
            )
            out_img.pixels[:] = gray_pixels

            # Save next to the reference image
            ref_path = bpy.path.abspath(outdoor.paint_reference_image)
            out_dir = os.path.dirname(ref_path) if ref_path else ""
            if not out_dir:
                out_dir = bpy.path.abspath("//")
            out_path = os.path.join(out_dir, "tileforge_painted_heightmap.png")

            out_img.filepath_raw = out_path
            out_img.file_format = 'PNG'
            out_img.save()
            bpy.data.images.remove(out_img)

            # Set as custom heightmap
            outdoor.heightmap_image = out_path
            outdoor.terrain_type = "CUSTOM"

            # Delete progress file — work is finalized
            progress = _paint_progress_path(outdoor)
            if progress and os.path.isfile(progress):
                os.remove(progress)

            # Clean up paint objects and images
            _remove_objects_by_prefix("TF_Paint_")
            mat = bpy.data.materials.get("TF_Paint_Material")
            if mat:
                bpy.data.materials.remove(mat)
            paint_img = bpy.data.images.get("TF_Paint_Heightmap")
            if paint_img:
                bpy.data.images.remove(paint_img)

            self.report({'INFO'}, f"Heightmap saved to {out_path}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to apply painted heightmap: {e}")
            return {'CANCELLED'}
        finally:
            outdoor.is_painting = False


class TILEFORGE_OT_CancelPaintMode(Operator):
    """Cancel painting and discard the painted heightmap"""
    bl_idname = "tileforge.cancel_paint_mode"
    bl_label = "Cancel"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.tile_forge.outdoor.is_painting

    def execute(self, context):
        outdoor = context.scene.tile_forge.outdoor

        try:
            # Exit paint mode
            if context.mode == 'PAINT_TEXTURE':
                bpy.ops.object.mode_set(mode='OBJECT')

            # Remove paint objects
            _remove_objects_by_prefix("TF_Paint_")
            mat = bpy.data.materials.get("TF_Paint_Material")
            if mat:
                bpy.data.materials.remove(mat)
            paint_img = bpy.data.images.get("TF_Paint_Heightmap")
            if paint_img:
                bpy.data.images.remove(paint_img)

            self.report({'INFO'}, "Paint mode cancelled")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to cancel paint mode: {e}")
            return {'CANCELLED'}
        finally:
            outdoor.is_painting = False


class TILEFORGE_OT_SavePaintProgress(Operator):
    """Save current paint progress to disk so you can resume later"""
    bl_idname = "tileforge.save_paint_progress"
    bl_label = "Save Progress"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return context.scene.tile_forge.outdoor.is_painting

    def execute(self, context):
        outdoor = context.scene.tile_forge.outdoor

        paint_img = bpy.data.images.get("TF_Paint_Heightmap")
        if not paint_img:
            self.report({'ERROR'}, "No painted heightmap to save")
            return {'CANCELLED'}

        progress_path = _paint_progress_path(outdoor)
        if not progress_path:
            self.report({'ERROR'}, "No reference image set — cannot determine save path")
            return {'CANCELLED'}

        paint_img.filepath_raw = progress_path
        paint_img.file_format = 'PNG'
        paint_img.save()

        self.report({'INFO'}, f"Progress saved to {progress_path}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Painted road mask
# ---------------------------------------------------------------------------

class TILEFORGE_OT_SetupRoadPaint(Operator):
    """Enter texture paint mode on the terrain to paint road paths"""
    bl_idname = "tileforge.setup_road_paint"
    bl_label = "Paint Roads"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        tf = context.scene.tile_forge
        outdoor = tf.outdoor
        tile = tf.tile

        if outdoor.is_painting:
            self.report({'ERROR'}, "Exit heightmap paint mode first")
            return {'CANCELLED'}

        # Terrain must exist
        obj = bpy.data.objects.get("TF_Preview_Terrain")
        if not obj:
            self.report({'ERROR'}, "Generate terrain first")
            return {'CANCELLED'}

        # Exit any current mode
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # --- Add UV layer with orthographic top-down projection ---
        mesh = obj.data
        half_x = m(tile.map_width, tile.print_scale) / 2.0
        half_y = m(tile.map_depth, tile.print_scale) / 2.0

        bm = bmesh.new()
        bm.from_mesh(mesh)
        # Remove existing road paint UV layer if present
        existing_uv = bm.loops.layers.uv.get("TF_RoadPaint_UV")
        if existing_uv:
            bm.loops.layers.uv.remove(existing_uv)
        uv_layer = bm.loops.layers.uv.new("TF_RoadPaint_UV")
        for face in bm.faces:
            for loop in face.loops:
                co = loop.vert.co
                loop[uv_layer].uv = (
                    (co.x + half_x) / (2.0 * half_x),
                    (co.y + half_y) / (2.0 * half_y),
                )
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        # --- Create or reuse paint image ---
        res = outdoor.road_paint_resolution
        paint_img = bpy.data.images.get("TF_RoadPaint_Image")
        if paint_img and (paint_img.size[0] != res or paint_img.size[1] != res):
            bpy.data.images.remove(paint_img)
            paint_img = None
        if not paint_img:
            paint_img = bpy.data.images.new(
                "TF_RoadPaint_Image", width=res, height=res,
                alpha=True, float_buffer=False,
            )
            paint_img.pixels[:] = [0.0] * (res * res * 4)

            # Load saved progress from disk if available
            progress_path = _road_paint_progress_path(outdoor)
            if progress_path and os.path.isfile(progress_path):
                saved = bpy.data.images.load(progress_path, check_existing=False)
                if saved.size[0] == res and saved.size[1] == res:
                    paint_img.pixels[:] = list(saved.pixels)
                bpy.data.images.remove(saved)
        paint_img.use_fake_user = True

        # --- Compute terrain Z range for height-based coloring ---
        min_z = float('inf')
        max_z = float('-inf')
        for v in mesh.vertices:
            z = v.co.z
            if z < min_z:
                min_z = z
            if z > max_z:
                max_z = z
        if min_z == max_z:
            max_z = min_z + 0.001

        # --- Build material ---
        mat = bpy.data.materials.new("TF_RoadPaint_Material")
        mat.use_nodes = True
        tree = mat.node_tree
        tree.nodes.clear()

        # Height-based terrain coloring
        n_texcoord = tree.nodes.new('ShaderNodeTexCoord')
        n_texcoord.location = (-800, 300)

        n_sep = tree.nodes.new('ShaderNodeSeparateXYZ')
        n_sep.location = (-600, 300)

        n_maprange = tree.nodes.new('ShaderNodeMapRange')
        n_maprange.inputs['From Min'].default_value = min_z
        n_maprange.inputs['From Max'].default_value = max_z
        n_maprange.inputs['To Min'].default_value = 0.0
        n_maprange.inputs['To Max'].default_value = 1.0
        n_maprange.location = (-400, 300)

        n_ramp = tree.nodes.new('ShaderNodeValToRGB')
        n_ramp.location = (-150, 300)
        # Green-to-brown terrain height visualization
        cr = n_ramp.color_ramp
        cr.elements[0].position = 0.0
        cr.elements[0].color = (0.2, 0.5, 0.15, 1.0)   # green lowland
        cr.elements[1].position = 1.0
        cr.elements[1].color = (0.45, 0.3, 0.15, 1.0)   # brown highland

        # Paint texture
        n_paint = tree.nodes.new('ShaderNodeTexImage')
        n_paint.name = "TF_PaintTex"
        n_paint.image = paint_img
        n_paint.location = (-400, -100)
        # Set UV map to our road paint UV layer
        n_uv = tree.nodes.new('ShaderNodeUVMap')
        n_uv.uv_map = "TF_RoadPaint_UV"
        n_uv.location = (-600, -100)

        # Opacity multiplier
        n_mult = tree.nodes.new('ShaderNodeMath')
        n_mult.name = "TF_OpacityMult"
        n_mult.operation = 'MULTIPLY'
        n_mult.inputs[1].default_value = outdoor.road_paint_overlay_opacity
        n_mult.location = (-100, -200)

        # Mix height color (A) with paint color (B)
        n_mix = tree.nodes.new('ShaderNodeMix')
        n_mix.name = "TF_MixColor"
        n_mix.data_type = 'RGBA'
        n_mix.location = (150, 200)

        n_bsdf = tree.nodes.new('ShaderNodeBsdfPrincipled')
        n_bsdf.location = (450, 200)

        n_out = tree.nodes.new('ShaderNodeOutputMaterial')
        n_out.location = (750, 200)

        # Links
        links = tree.links
        # Height coloring chain
        links.new(n_texcoord.outputs['Generated'], n_sep.inputs['Vector'])
        links.new(n_sep.outputs['Z'], n_maprange.inputs['Value'])
        links.new(n_maprange.outputs['Result'], n_ramp.inputs['Fac'])
        # Paint texture UV
        links.new(n_uv.outputs['UV'], n_paint.inputs['Vector'])
        # Opacity
        links.new(n_paint.outputs['Alpha'], n_mult.inputs[0])
        links.new(n_mult.outputs['Value'], n_mix.inputs['Factor'])
        # Mix: A = height color, B = paint color
        links.new(n_ramp.outputs['Color'], n_mix.inputs[6])
        links.new(n_paint.outputs['Color'], n_mix.inputs[7])
        # Output
        links.new(n_mix.outputs[2], n_bsdf.inputs['Base Color'])
        links.new(n_bsdf.outputs['BSDF'], n_out.inputs['Surface'])

        # Set paint texture as active for texture paint mode
        tree.nodes.active = n_paint

        # --- Store original material and assign paint material ---
        if obj.data.materials:
            obj["TF_OrigMaterial"] = obj.data.materials[0].name
        else:
            obj["TF_OrigMaterial"] = ""
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        # Select terrain and enter texture paint mode
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='TEXTURE_PAINT')

        # Configure brush for white (roads are binary — painted or not)
        ip = context.tool_settings.image_paint
        brush = ip.brush
        if brush:
            brush.color = (1.0, 1.0, 1.0)
            brush.blend = 'MIX'
            ups = ip.unified_paint_settings
            if ups.use_unified_color:
                ups.color = (1.0, 1.0, 1.0)

        # Switch viewport to Material Preview
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'MATERIAL'
                        break
                break

        outdoor.is_road_painting = True
        self.report({'INFO'}, "Paint road paths on terrain — brush size controls road width")
        return {'FINISHED'}


def _restore_terrain_after_road_paint(obj):
    """Restore terrain object after road painting — shared by Apply and Cancel."""
    # Restore original material
    orig_name = obj.get("TF_OrigMaterial", "")
    if orig_name:
        orig_mat = bpy.data.materials.get(orig_name)
        if orig_mat and obj.data.materials:
            obj.data.materials[0] = orig_mat
    elif obj.data.materials:
        # No original material stored — clear the slot
        obj.data.materials.pop(index=0)

    # Remove custom property
    if "TF_OrigMaterial" in obj:
        del obj["TF_OrigMaterial"]

    # Remove paint material
    mat = bpy.data.materials.get("TF_RoadPaint_Material")
    if mat:
        bpy.data.materials.remove(mat)

    # Remove road paint UV layer
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    uv = bm.loops.layers.uv.get("TF_RoadPaint_UV")
    if uv:
        bm.loops.layers.uv.remove(uv)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    # Remove paint image
    paint_img = bpy.data.images.get("TF_RoadPaint_Image")
    if paint_img:
        bpy.data.images.remove(paint_img)


class TILEFORGE_OT_ApplyPaintedRoad(Operator):
    """Convert the painted overlay to a road mask and enable painted roads"""
    bl_idname = "tileforge.apply_painted_road"
    bl_label = "Apply Road Mask"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.tile_forge.outdoor.is_road_painting

    def execute(self, context):
        outdoor = context.scene.tile_forge.outdoor

        try:
            # Exit paint mode
            if context.mode == 'PAINT_TEXTURE':
                bpy.ops.object.mode_set(mode='OBJECT')

            paint_img = bpy.data.images.get("TF_RoadPaint_Image")
            if not paint_img:
                self.report({'ERROR'}, "No painted road image found")
                return {'CANCELLED'}

            # Copy paint image to persistent mask
            w, h = paint_img.size
            mask_img = bpy.data.images.get("TF_RoadPaint_Mask")
            if mask_img and (mask_img.size[0] != w or mask_img.size[1] != h):
                bpy.data.images.remove(mask_img)
                mask_img = None
            if not mask_img:
                mask_img = bpy.data.images.new(
                    "TF_RoadPaint_Mask", width=w, height=h,
                    alpha=True, float_buffer=False,
                )
            mask_img.pixels[:] = list(paint_img.pixels)
            mask_img.use_fake_user = True

            # Enable painted road in generation pipeline
            outdoor.add_painted_road = True

            # Delete progress file — work is finalized
            progress = _road_paint_progress_path(outdoor)
            if progress and os.path.isfile(progress):
                os.remove(progress)

            # Restore terrain
            obj = bpy.data.objects.get("TF_Preview_Terrain")
            if obj:
                _restore_terrain_after_road_paint(obj)

            self.report({'INFO'}, "Road mask applied — regenerate terrain to see roads")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to apply road mask: {e}")
            return {'CANCELLED'}
        finally:
            outdoor.is_road_painting = False


class TILEFORGE_OT_CancelRoadPaint(Operator):
    """Cancel road painting and discard the painted roads"""
    bl_idname = "tileforge.cancel_road_paint"
    bl_label = "Cancel"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.tile_forge.outdoor.is_road_painting

    def execute(self, context):
        outdoor = context.scene.tile_forge.outdoor

        try:
            # Exit paint mode
            if context.mode == 'PAINT_TEXTURE':
                bpy.ops.object.mode_set(mode='OBJECT')

            # Restore terrain
            obj = bpy.data.objects.get("TF_Preview_Terrain")
            if obj:
                _restore_terrain_after_road_paint(obj)

            self.report({'INFO'}, "Road paint mode cancelled")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to cancel road paint mode: {e}")
            return {'CANCELLED'}
        finally:
            outdoor.is_road_painting = False


class TILEFORGE_OT_SaveRoadPaintProgress(Operator):
    """Save current road paint progress to disk so you can resume later"""
    bl_idname = "tileforge.save_road_paint_progress"
    bl_label = "Save Progress"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return context.scene.tile_forge.outdoor.is_road_painting

    def execute(self, context):
        outdoor = context.scene.tile_forge.outdoor

        paint_img = bpy.data.images.get("TF_RoadPaint_Image")
        if not paint_img:
            self.report({'ERROR'}, "No painted road image to save")
            return {'CANCELLED'}

        progress_path = _road_paint_progress_path(outdoor)
        if not progress_path:
            self.report({'ERROR'}, "Save .blend file first to determine save path")
            return {'CANCELLED'}

        paint_img.filepath_raw = progress_path
        paint_img.file_format = 'PNG'
        paint_img.save()

        self.report({'INFO'}, f"Road paint progress saved to {progress_path}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# AI Heightmap generation
# ---------------------------------------------------------------------------

class TILEFORGE_OT_GenerateAIHeightmap(Operator):
    """Send a color map to an AI service and receive a grayscale heightmap"""
    bl_idname = "tileforge.generate_ai_heightmap"
    bl_label = "Generate AI Heightmap"
    bl_options = {'REGISTER'}

    def execute(self, context):
        outdoor = context.scene.tile_forge.outdoor

        # Resolve API key from addon preferences
        prefs = context.preferences.addons[__package__].preferences
        provider = outdoor.ai_provider
        if provider == "OPENAI":
            api_key = prefs.openai_api_key.strip()
        else:
            api_key = prefs.gemini_api_key.strip()

        image_path = bpy.path.abspath(outdoor.ai_source_image)

        context.window.cursor_set('WAIT')
        try:
            out_path = ai_api.generate_heightmap(
                provider, api_key, image_path, outdoor.ai_custom_prompt,
            )
        except (ValueError, ConnectionError) as e:
            context.window.cursor_set('DEFAULT')
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        context.window.cursor_set('DEFAULT')

        # Wire result into the Custom Heightmap pipeline
        outdoor.heightmap_image = out_path
        outdoor.terrain_type = "CUSTOM"

        self.report({'INFO'}, f"AI heightmap saved to {out_path}")
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
        # Exit paint mode before removing objects
        outdoor = context.scene.tile_forge.outdoor
        if outdoor.is_painting or outdoor.is_road_painting:
            if context.mode == 'PAINT_TEXTURE':
                bpy.ops.object.mode_set(mode='OBJECT')
            outdoor.is_painting = False
            outdoor.is_road_painting = False

        count = 0
        count += _remove_objects_by_prefix("TF_Preview")
        count += _remove_objects_by_prefix("tile_r")
        count += _remove_objects_by_prefix("TF_Paint_")
        count += _remove_objects_by_prefix("TF_Road_")
        count += _remove_objects_by_prefix("TF_RoadPaint_")
        mat = bpy.data.materials.get("TF_Paint_Material")
        if mat:
            bpy.data.materials.remove(mat)
        mat = bpy.data.materials.get("TF_RoadPaint_Material")
        if mat:
            bpy.data.materials.remove(mat)

        # Remove paint-related images
        for img_name in ("TF_Paint_Heightmap", "TF_RoadPaint_Image", "TF_RoadPaint_Mask"):
            img = bpy.data.images.get(img_name)
            if img:
                bpy.data.images.remove(img)

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


def _paint_progress_path(outdoor):
    """Return deterministic path for paint progress file next to the reference image."""
    ref = bpy.path.abspath(outdoor.paint_reference_image)
    if not ref:
        return ""
    return os.path.join(os.path.dirname(ref), "tileforge_paint_progress.png")


def _road_paint_progress_path(outdoor):
    """Return deterministic path for road paint progress file next to the .blend file."""
    blend = bpy.data.filepath
    if not blend:
        return ""
    return os.path.join(os.path.dirname(blend), "tileforge_road_paint_progress.png")


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
    TILEFORGE_OT_AddColorZone,
    TILEFORGE_OT_RemoveColorZone,
    TILEFORGE_OT_AddRoadSegment,
    TILEFORGE_OT_RemoveRoadSegment,
    TILEFORGE_OT_SetupHeightmapPaint,
    TILEFORGE_OT_SetBrushHeight,
    TILEFORGE_OT_ApplyPaintedHeightmap,
    TILEFORGE_OT_CancelPaintMode,
    TILEFORGE_OT_SavePaintProgress,
    TILEFORGE_OT_SetupRoadPaint,
    TILEFORGE_OT_ApplyPaintedRoad,
    TILEFORGE_OT_CancelRoadPaint,
    TILEFORGE_OT_SaveRoadPaintProgress,
    TILEFORGE_OT_GenerateAIHeightmap,
    TILEFORGE_OT_CleanupAll,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
