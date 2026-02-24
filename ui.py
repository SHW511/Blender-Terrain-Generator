"""
UI Panels for D&D Tile Forge.
Organized as a sidebar panel in the 3D Viewport under the 'Tile Forge' tab.
"""

import math

import bpy
from bpy.types import Panel


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TILEFORGE_PT_Base:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tile Forge'


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class TILEFORGE_PT_Main(TILEFORGE_PT_Base, Panel):
    bl_idname = "TILEFORGE_PT_main"
    bl_label = "D&D Tile Forge"

    def draw(self, context):
        layout = self.layout
        tf = context.scene.tile_forge

        layout.prop(tf, "generation_mode", expand=True)


# ---------------------------------------------------------------------------
# Tile Settings
# ---------------------------------------------------------------------------

class TILEFORGE_PT_TileSettings(TILEFORGE_PT_Base, Panel):
    bl_idname = "TILEFORGE_PT_tile_settings"
    bl_label = "Tile Settings"
    bl_parent_id = "TILEFORGE_PT_main"

    def draw(self, context):
        layout = self.layout
        tile = context.scene.tile_forge.tile

        layout.use_property_split = True
        layout.use_property_decorate = False

        # Print scale
        layout.prop(tile, "print_scale")

        layout.separator()

        col = layout.column(align=True)
        col.prop(tile, "map_width")
        col.prop(tile, "map_depth")

        layout.separator()

        col = layout.column(align=True)
        col.prop(tile, "tile_size_x")
        col.prop(tile, "tile_size_y")
        col.prop(tile, "base_height")

        # Show tile count that slicing will produce
        cols = max(1, math.floor(tile.map_width / tile.tile_size_x))
        rows = max(1, math.floor(tile.map_depth / tile.tile_size_y))
        box = layout.box()
        box.label(
            text=f"Map: {tile.map_width:.1f} x {tile.map_depth:.1f} m "
                 f"({cols} x {rows} = {cols * rows} tiles)",
            icon='GRID',
        )

        layout.separator()

        # Grid lines
        col = layout.column(align=True)
        col.prop(tile, "engrave_grid")
        if tile.engrave_grid:
            col.prop(tile, "grid_preset")
            if tile.grid_preset == 'CUSTOM':
                col.prop(tile, "grid_squares")
            col.prop(tile, "grid_line_depth")
            col.prop(tile, "grid_line_width")

        layout.separator()

        # Print dimensions info
        ps = tile.print_scale
        tile_mm_x = tile.tile_size_x / ps * 1000
        tile_mm_y = tile.tile_size_y / ps * 1000

        _preset_sizes = {"1_INCH": 25.4, "25MM": 25.0}
        if tile.grid_preset in _preset_sizes:
            grid_mm_x = _preset_sizes[tile.grid_preset]
            grid_mm_y = grid_mm_x
        else:
            grid_sq = tile.grid_squares
            grid_mm_x = tile_mm_x / grid_sq
            grid_sq_y = max(1, round(tile.tile_size_y / (tile.tile_size_x / grid_sq)))
            grid_mm_y = tile_mm_y / grid_sq_y

        map_mm_x = tile.map_width / ps * 1000
        map_mm_y = tile.map_depth / ps * 1000

        box = layout.box()
        box.label(text="Print Dimensions", icon='DRIVER_DISTANCE')
        box.label(text=f"  Tile: {tile_mm_x:.1f} x {tile_mm_y:.1f} mm")
        box.label(text=f"  Grid square: {grid_mm_x:.1f} x {grid_mm_y:.1f} mm")
        box.label(text=f"  Map total: {map_mm_x:.1f} x {map_mm_y:.1f} mm")

        layout.separator()

        # Connectors
        col = layout.column(align=True)
        col.prop(tile, "connector_type")
        if tile.connector_type != 'NONE':
            col.prop(tile, "connector_diameter")
            col.prop(tile, "connector_height")
            col.prop(tile, "connector_tolerance")


# ---------------------------------------------------------------------------
# Outdoor Terrain
# ---------------------------------------------------------------------------

class TILEFORGE_PT_OutdoorTerrain(TILEFORGE_PT_Base, Panel):
    bl_idname = "TILEFORGE_PT_outdoor"
    bl_label = "Outdoor Terrain"
    bl_parent_id = "TILEFORGE_PT_main"

    @classmethod
    def poll(cls, context):
        return context.scene.tile_forge.generation_mode == 'OUTDOOR'

    def draw(self, context):
        layout = self.layout
        outdoor = context.scene.tile_forge.outdoor

        layout.use_property_split = True
        layout.use_property_decorate = False

        col = layout.column(align=True)
        col.prop(outdoor, "terrain_type")

        if outdoor.terrain_type == "CUSTOM":
            col.prop(outdoor, "heightmap_image")

            # Crop margins
            crop_box = col.box()
            crop_box.label(text="Crop Margins", icon='FULLSCREEN_EXIT')
            row = crop_box.row(align=True)
            row.prop(outdoor, "crop_left", text="L")
            row.prop(outdoor, "crop_right", text="R")
            row = crop_box.row(align=True)
            row.prop(outdoor, "crop_top", text="T")
            row.prop(outdoor, "crop_bottom", text="B")

        elif outdoor.terrain_type == "MAP_IMAGE":
            col.prop(outdoor, "heightmap_image")

            # Crop margins
            crop_box = col.box()
            crop_box.label(text="Crop Margins", icon='FULLSCREEN_EXIT')
            row = crop_box.row(align=True)
            row.prop(outdoor, "crop_left", text="L")
            row.prop(outdoor, "crop_right", text="R")
            row = crop_box.row(align=True)
            row.prop(outdoor, "crop_top", text="T")
            row.prop(outdoor, "crop_bottom", text="B")

            col.separator()

            # Color zones
            zone_box = col.box()
            zone_box.label(text="Color Zones", icon='COLOR')
            for i, zone in enumerate(outdoor.color_zones):
                row = zone_box.row(align=True)
                row.prop(zone, "color", text="")
                row.prop(zone, "zone_name", text="")
                row.prop(zone, "height", text="H", slider=True)
                row.prop(zone, "tolerance", text="T", slider=True)
            row = zone_box.row(align=True)
            row.operator("tileforge.add_color_zone", text="Add Zone", icon='ADD')
            row.operator("tileforge.remove_color_zone", text="Remove", icon='REMOVE')

            col.separator()
            col.prop(outdoor, "map_smoothing")
            col.prop(outdoor, "edge_preserve_strength")
            col.prop(outdoor, "fallback_height")

        else:
            col.prop(outdoor, "noise_type")
            if outdoor.noise_type != 'VORONOI':
                col.prop(outdoor, "noise_basis")
            col.prop(outdoor, "noise_scale")
            col.prop(outdoor, "noise_detail")
            col.prop(outdoor, "noise_roughness")
            col.prop(outdoor, "lacunarity")

            # Show offset/gain only for noise types that use them
            if outdoor.noise_type in {'HETERO_TERRAIN', 'RIDGED', 'HYBRID'}:
                col.prop(outdoor, "fractal_offset")
            if outdoor.noise_type in {'RIDGED', 'HYBRID'}:
                col.prop(outdoor, "fractal_gain")

            row = col.row(align=True)
            row.prop(outdoor, "noise_seed")
            row.operator("tileforge.randomize_seed", text="", icon='FILE_REFRESH')

        layout.separator()

        # Paint Heightmap tool
        # Safety check: reset is_painting if paint plane was removed externally
        if outdoor.is_painting:
            has_paint_obj = any(
                obj.name.startswith("TF_Paint_") for obj in bpy.data.objects
            )
            if not has_paint_obj:
                outdoor.is_painting = False

        box = layout.box()
        box.label(text="Paint Heightmap", icon='BRUSH_DATA')

        if not outdoor.is_painting:
            col = box.column(align=True)
            col.prop(outdoor, "paint_reference_image")
            col.prop(outdoor, "paint_resolution")
            col.operator(
                "tileforge.setup_heightmap_paint",
                text="Start Painting",
                icon='BRUSHES_ALL',
            )
        else:
            col = box.column(align=True)
            col.prop(outdoor, "paint_overlay_opacity", slider=True)

            col.label(text="Height Levels:")
            row = col.row(align=True)
            op = row.operator("tileforge.set_brush_height", text="Sea")
            op.level = 'SEA'
            op = row.operator("tileforge.set_brush_height", text="Low")
            op.level = 'LOW'
            op = row.operator("tileforge.set_brush_height", text="Mid")
            op.level = 'MID'
            row = col.row(align=True)
            op = row.operator("tileforge.set_brush_height", text="High")
            op.level = 'HIGH'
            op = row.operator("tileforge.set_brush_height", text="Peak")
            op.level = 'PEAK'
            op = row.operator("tileforge.set_brush_height", text="Eraser")
            op.level = 'ERASER'

            col.separator()
            row = col.row(align=True)
            row.operator(
                "tileforge.apply_painted_heightmap",
                text="Apply Heightmap",
                icon='CHECKMARK',
            )
            row.operator(
                "tileforge.cancel_paint_mode",
                text="Cancel",
                icon='X',
            )

        layout.separator()

        col = layout.column(align=True)
        col.prop(outdoor, "terrain_height_min")
        col.prop(outdoor, "terrain_height_max")
        col.prop(outdoor, "height_exponent")
        col.prop(outdoor, "subdivisions")

        # Show physical relief range
        tile = context.scene.tile_forge.tile
        ps = tile.print_scale
        relief_min_mm = outdoor.terrain_height_min / ps * 1000
        relief_max_mm = outdoor.terrain_height_max / ps * 1000
        base_mm = tile.base_height
        col.label(text=f"  Relief: {relief_min_mm:.1f} \u2013 {relief_max_mm:.1f} mm (+ {base_mm:.1f} mm base)")

        layout.separator()

        # Terrain Shaping
        box = layout.box()
        box.label(text="Terrain Shaping", icon='MOD_SMOOTH')

        # Domain warping (hidden for image-based modes)
        if outdoor.terrain_type not in ("CUSTOM", "MAP_IMAGE"):
            col = box.column(align=True)
            col.prop(outdoor, "enable_domain_warp")
            if outdoor.enable_domain_warp:
                col.prop(outdoor, "warp_strength")
                col.prop(outdoor, "warp_scale")

            box.separator()

        # Terracing (available for procedural and color-map modes)
        if outdoor.terrain_type != "CUSTOM":
            col = box.column(align=True)
            col.prop(outdoor, "enable_terracing")
            if outdoor.enable_terracing:
                col.prop(outdoor, "terrace_levels")
                col.prop(outdoor, "terrace_sharpness")

            box.separator()

        # Slope clamping (always visible — works for heightmaps too)
        col = box.column(align=True)
        col.prop(outdoor, "enable_slope_clamp")
        if outdoor.enable_slope_clamp:
            col.prop(outdoor, "max_slope_angle")

        layout.separator()

        # Noise Layers (hidden for image-based modes)
        if outdoor.terrain_type not in ("CUSTOM", "MAP_IMAGE"):
            box = layout.box()
            box.label(text="Noise Layers", icon='NODETREE')

            col = box.column(align=True)
            col.prop(outdoor, "enable_noise_layers")

            if outdoor.enable_noise_layers:
                for i, layer in enumerate(outdoor.noise_layers):
                    layer_box = box.box()
                    header = layer_box.row(align=True)
                    header.prop(layer, "enabled", text="")
                    header.prop(layer, "layer_name", text="")
                    header.label(text=f"#{i + 1}")

                    if layer.enabled:
                        col2 = layer_box.column(align=True)
                        col2.prop(layer, "noise_type")
                        if layer.noise_type != 'VORONOI':
                            col2.prop(layer, "noise_basis")
                        col2.prop(layer, "scale")
                        col2.prop(layer, "strength")
                        col2.prop(layer, "octaves")
                        col2.prop(layer, "blend_mode")
                        col2.prop(layer, "seed_offset")

                        col2.separator()
                        col2.prop(layer, "use_mask")
                        if layer.use_mask:
                            col2.prop(layer, "mask_invert")
                            col2.prop(layer, "mask_contrast")

                row = box.row(align=True)
                row.operator("tileforge.add_noise_layer", text="Add Layer", icon='ADD')
                row.operator("tileforge.remove_noise_layer", text="Remove", icon='REMOVE')

            layout.separator()

        # Erosion
        box = layout.box()
        box.label(text="Erosion", icon='FORCE_WIND')

        # Hydraulic erosion
        col = box.column(align=True)
        col.prop(outdoor, "enable_hydraulic_erosion")
        if outdoor.enable_hydraulic_erosion:
            col.prop(outdoor, "hydraulic_preset")
            col.prop(outdoor, "hydraulic_show_advanced")
            if outdoor.hydraulic_show_advanced:
                col.prop(outdoor, "hydraulic_droplets")
                col.prop(outdoor, "hydraulic_erosion_rate")
                col.prop(outdoor, "hydraulic_deposition_rate")
                col.prop(outdoor, "hydraulic_evaporation")
                col.prop(outdoor, "hydraulic_inertia")
                col.prop(outdoor, "hydraulic_capacity_mult")
                col.prop(outdoor, "hydraulic_min_slope")
                col.prop(outdoor, "hydraulic_radius")

        box.separator()

        # Thermal erosion
        col = box.column(align=True)
        col.prop(outdoor, "enable_thermal_erosion")
        if outdoor.enable_thermal_erosion:
            col.prop(outdoor, "thermal_talus_angle")
            col.prop(outdoor, "thermal_iterations")
            col.prop(outdoor, "thermal_strength")

        layout.separator()

        # Ground texture
        col = layout.column(align=True)
        col.prop(outdoor, "floor_texture")
        if outdoor.floor_texture != 'NONE':
            col.prop(outdoor, "texture_strength")

        layout.separator()

        # Terrain features
        box = layout.box()
        box.label(text="Terrain Features", icon='MESH_PLANE')

        col = box.column(align=True)
        col.prop(outdoor, "add_river")
        if outdoor.add_river:
            col.prop(outdoor, "river_curve")
            col.prop(outdoor, "river_width")
            col.prop(outdoor, "river_depth")
            col.prop(outdoor, "river_meander_strength")

        box.separator()

        col = box.column(align=True)
        col.prop(outdoor, "add_path")
        if outdoor.add_path:
            col.prop(outdoor, "path_curve")
            col.prop(outdoor, "path_width")

        layout.separator()

        # Generate button
        layout.operator(
            "tileforge.generate_terrain",
            text="Generate Terrain",
            icon='RNDCURVE',
        )


# ---------------------------------------------------------------------------
# Dungeon Settings
# ---------------------------------------------------------------------------

class TILEFORGE_PT_Dungeon(TILEFORGE_PT_Base, Panel):
    bl_idname = "TILEFORGE_PT_dungeon"
    bl_label = "Dungeon / Interior"
    bl_parent_id = "TILEFORGE_PT_main"

    @classmethod
    def poll(cls, context):
        return context.scene.tile_forge.generation_mode == 'DUNGEON'

    def draw(self, context):
        layout = self.layout
        dungeon = context.scene.tile_forge.dungeon

        layout.use_property_split = True
        layout.use_property_decorate = False

        col = layout.column(align=True)
        col.prop(dungeon, "dungeon_source")

        if dungeon.dungeon_source == "IMAGE":
            col.prop(dungeon, "floorplan_image")
        else:
            box = layout.box()
            box.label(text="Grid editor coming soon", icon='INFO')

        layout.separator()

        col = layout.column(align=True)
        col.prop(dungeon, "wall_height")
        col.prop(dungeon, "wall_thickness")

        layout.separator()

        col = layout.column(align=True)
        col.prop(dungeon, "doorway_width")
        col.prop(dungeon, "doorway_height")

        layout.separator()

        col = layout.column(align=True)
        col.prop(dungeon, "floor_texture")
        if dungeon.floor_texture != 'NONE':
            col.prop(dungeon, "texture_strength")

        layout.separator()

        # Stairs
        box = layout.box()
        box.label(text="Elevation", icon='MOD_ARRAY')
        col = box.column(align=True)
        col.prop(dungeon, "add_stairs")
        if dungeon.add_stairs:
            col.prop(dungeon, "stair_step_height")
            col.prop(dungeon, "stair_step_depth")

        layout.separator()

        layout.operator(
            "tileforge.generate_dungeon",
            text="Generate Dungeon",
            icon='MOD_BUILD',
        )


# ---------------------------------------------------------------------------
# Slicing & Export
# ---------------------------------------------------------------------------

class TILEFORGE_PT_SliceExport(TILEFORGE_PT_Base, Panel):
    bl_idname = "TILEFORGE_PT_slice_export"
    bl_label = "Slice & Export"
    bl_parent_id = "TILEFORGE_PT_main"

    def draw(self, context):
        layout = self.layout
        tf = context.scene.tile_forge
        export = tf.export

        layout.use_property_split = True
        layout.use_property_decorate = False

        # Tile grid for slicing
        tile = tf.tile
        col = layout.column(align=True)
        col.prop(tile, "map_tiles_x")
        col.prop(tile, "map_tiles_y")

        layout.separator()

        # Slice
        layout.operator(
            "tileforge.slice_tiles",
            text="Slice into Tiles",
            icon='MESH_GRID',
        )

        layout.separator()

        # Export settings
        col = layout.column(align=True)
        col.prop(export, "export_path")
        col.prop(export, "filename_prefix")

        layout.separator()

        col = layout.column(align=True)
        col.prop(export, "check_manifold")
        col.prop(export, "check_thickness")
        if export.check_thickness:
            col.prop(export, "min_wall_thickness")

        layout.separator()

        row = layout.row(align=True)
        row.operator(
            "tileforge.export_tiles",
            text="Export All Tiles",
            icon='EXPORT',
        )
        row.operator(
            "tileforge.export_single_tile",
            text="Export Selected",
            icon='OBJECT_DATA',
        )

        layout.separator()

        layout.operator(
            "tileforge.cleanup_all",
            text="Clean Up All",
            icon='TRASH',
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = (
    TILEFORGE_PT_Main,
    TILEFORGE_PT_TileSettings,
    TILEFORGE_PT_OutdoorTerrain,
    TILEFORGE_PT_Dungeon,
    TILEFORGE_PT_SliceExport,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
