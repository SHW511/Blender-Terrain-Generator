"""
Property definitions for D&D Tile Forge.
All user-configurable settings are defined here as Blender property groups.

World dimensions (tile size, map size, terrain height, etc.) are in meters
at game-world scale. Physical print dimensions (base height, grid line depth,
connector tolerances, etc.) are in millimeters.
"""

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

GENERATION_MODES = [
    ("OUTDOOR", "Outdoor Terrain", "Generate outdoor terrain from heightmap or procedural noise"),
    ("DUNGEON", "Dungeon / Interior", "Generate dungeon tiles from a floor plan image or grid"),
]

TERRAIN_TYPES = [
    ("FLAT", "Flat / Grassland", "Subtle low-amplitude noise"),
    ("ROCKY", "Rocky / Mountain", "High-frequency rocky terrain"),
    ("DESERT", "Desert Dunes", "Smooth rolling dune shapes"),
    ("FOREST", "Forest Floor", "Organic bumpy forest ground"),
    ("CUSTOM", "Custom Heightmap", "Load a grayscale image as heightmap"),
]

NOISE_TYPES = [
    ("PERLIN", "Perlin (fBm)", "Classic fractional Brownian motion"),
    ("VORONOI", "Voronoi", "Cell-based Voronoi noise"),
    ("MULTI_FRACTAL", "Multi-Fractal", "Multiplicative fractal — natural flat/rough variation"),
    ("HETERO_TERRAIN", "Hetero Terrain", "Altitude-dependent detail — smooth valleys, rough peaks"),
    ("RIDGED", "Ridged Multi-Fractal", "Sharp mountain ridges and canyon walls"),
    ("HYBRID", "Hybrid Multi-Fractal", "Best general-purpose terrain — blends fBm with multi-fractal"),
]

NOISE_BASES = [
    ("PERLIN_ORIGINAL", "Perlin Classic", "Classic Perlin noise basis"),
    ("PERLIN_NEW", "Perlin Improved", "Smoother improved Perlin noise"),
    ("BLENDER", "Blender", "Blender's built-in noise"),
    ("VORONOI_F1", "Voronoi F1", "Smooth rounded cells"),
    ("VORONOI_F2F1", "Voronoi Cracked", "Sharp cell edges — cracked earth"),
    ("VORONOI_CRACKLE", "Voronoi Crackle", "Crystalline crack patterns"),
    ("CELLNOISE", "Cell Noise", "Flat-shaded cells"),
]

DUNGEON_SOURCE = [
    ("IMAGE", "Floor Plan Image", "Import a black/white floor plan image"),
    ("GRID", "Grid Editor", "Draw walls on a grid (future feature)"),
]

CONNECTOR_TYPES = [
    ("NONE", "None", "No connectors"),
    ("PEG_SLOT", "Peg & Slot", "Cylindrical pegs on two edges, matching slots on opposite edges"),
    ("TONGUE_GROOVE", "Tongue & Groove", "Raised ridge on two edges, matching channel on opposite edges"),
]

FLOOR_TEXTURES = [
    ("NONE", "Smooth", "No floor texture"),
    ("STONE_SLAB", "Stone Slab", "Rectangular stone slab pattern"),
    ("COBBLESTONE", "Cobblestone", "Irregular cobblestone pattern"),
    ("WOOD_PLANK", "Wood Plank", "Parallel wood plank lines"),
    ("DIRT", "Dirt / Earth", "Rough organic ground texture"),
]

NOISE_BLEND_MODES = [
    ("ADD", "Add", "Add layer height to base"),
    ("MULTIPLY", "Multiply", "Multiply base height by layer value"),
    ("SCREEN", "Screen", "Lighten — inverse multiply for subtle peaks"),
    ("OVERLAY", "Overlay", "Contrast-enhance — darken lows, lighten highs"),
]

HYDRAULIC_PRESETS = [
    ("LIGHT", "Light", "Subtle weathering — soft channels"),
    ("MEDIUM", "Medium", "Balanced erosion — visible valleys"),
    ("HEAVY", "Heavy", "Aggressive carving — deep gorges"),
]


# ---------------------------------------------------------------------------
# Property Groups
# ---------------------------------------------------------------------------

class TILEFORGE_PG_TileSettings(PropertyGroup):
    """Core tile dimensions and grid settings."""

    print_scale: IntProperty(
        name="Print Scale (1:N)",
        description="World-to-print scale ratio. Standard D&D 28mm scale is 1:60",
        default=60,
        min=1,
        max=500,
    )

    tile_size_x: FloatProperty(
        name="Tile Width (m)",
        description="Width of each tile in world meters",
        default=6.0,
        min=1.5,
        max=18.0,
        step=50,
        precision=1,
    )
    tile_size_y: FloatProperty(
        name="Tile Depth (m)",
        description="Depth of each tile in world meters",
        default=6.0,
        min=1.5,
        max=18.0,
        step=50,
        precision=1,
    )
    base_height: FloatProperty(
        name="Base Height (mm)",
        description="Solid base thickness under terrain",
        default=3.0,
        min=1.0,
        max=10.0,
        step=50,
        precision=1,
    )
    grid_squares: IntProperty(
        name="Grid Squares per Tile",
        description="Number of 5-foot (1.5m) squares along one tile edge",
        default=4,
        min=1,
        max=12,
    )
    engrave_grid: BoolProperty(
        name="Engrave Grid Lines",
        description="Cut 5-foot (1.5m) grid lines into the tile surface",
        default=True,
    )
    grid_line_depth: FloatProperty(
        name="Grid Line Depth (mm)",
        description="How deep grid lines are engraved",
        default=0.4,
        min=0.1,
        max=1.0,
        step=10,
        precision=2,
    )
    grid_line_width: FloatProperty(
        name="Grid Line Width (mm)",
        description="Width of engraved grid lines",
        default=0.6,
        min=0.2,
        max=2.0,
        step=10,
        precision=2,
    )

    # Map layout — total terrain size
    map_width: FloatProperty(
        name="Map Width (m)",
        description="Total terrain width in world meters",
        default=18.0,
        min=3.0,
        max=120.0,
        step=100,
        precision=1,
    )
    map_depth: FloatProperty(
        name="Map Depth (m)",
        description="Total terrain depth in world meters",
        default=18.0,
        min=3.0,
        max=120.0,
        step=100,
        precision=1,
    )

    # Tile grid for slicing
    map_tiles_x: IntProperty(
        name="Map Columns",
        description="Number of tiles across the map (X axis)",
        default=3,
        min=1,
        max=20,
    )
    map_tiles_y: IntProperty(
        name="Map Rows",
        description="Number of tiles down the map (Y axis)",
        default=3,
        min=1,
        max=20,
    )

    # Connectors
    connector_type: EnumProperty(
        name="Connector Type",
        items=CONNECTOR_TYPES,
        default="PEG_SLOT",
    )
    connector_tolerance: FloatProperty(
        name="Connector Tolerance (mm)",
        description="Gap between peg and slot for print tolerance",
        default=0.3,
        min=0.1,
        max=0.8,
        step=5,
        precision=2,
    )
    connector_diameter: FloatProperty(
        name="Connector Diameter (mm)",
        description="Diameter of peg connectors",
        default=4.0,
        min=2.0,
        max=8.0,
        step=50,
        precision=1,
    )
    connector_height: FloatProperty(
        name="Connector Height (mm)",
        description="How tall the peg extends / how deep the slot is",
        default=3.0,
        min=1.5,
        max=6.0,
        step=50,
        precision=1,
    )


class TILEFORGE_PG_NoiseLayer(PropertyGroup):
    """A single additional noise layer for multi-layer composition."""

    enabled: BoolProperty(
        name="Enabled",
        description="Enable this noise layer",
        default=True,
    )
    layer_name: StringProperty(
        name="Name",
        description="Display name for this layer",
        default="Detail",
    )
    noise_type: EnumProperty(
        name="Noise Type",
        items=NOISE_TYPES,
        default="PERLIN",
    )
    noise_basis: EnumProperty(
        name="Noise Basis",
        items=NOISE_BASES,
        default="PERLIN_NEW",
    )
    scale: FloatProperty(
        name="Scale",
        description="Noise scale for this layer",
        default=6.0,
        min=0.1,
        max=50.0,
        step=50,
        precision=2,
    )
    strength: FloatProperty(
        name="Strength (m)",
        description="Layer height contribution in world meters",
        default=0.06,
        min=0.0,
        max=0.6,
        step=1,
        precision=2,
    )
    octaves: IntProperty(
        name="Octaves",
        description="Number of noise octaves",
        default=3,
        min=1,
        max=8,
    )
    blend_mode: EnumProperty(
        name="Blend Mode",
        items=NOISE_BLEND_MODES,
        default="ADD",
    )
    seed_offset: IntProperty(
        name="Seed Offset",
        description="Offset added to base seed for this layer",
        default=1000,
    )
    use_mask: BoolProperty(
        name="Height Mask",
        description="Mask layer effect by vertex height",
        default=False,
    )
    mask_invert: BoolProperty(
        name="Invert Mask",
        description="Invert the height mask (apply to valleys instead of peaks)",
        default=False,
    )
    mask_contrast: FloatProperty(
        name="Mask Contrast",
        description="Power curve for mask sharpness",
        default=2.0,
        min=0.5,
        max=10.0,
        step=10,
        precision=1,
    )


def _update_hydraulic_preset(self, context):
    """Set advanced hydraulic params from preset selection."""
    presets = {
        "LIGHT": (1000, 0.15, 0.4, 0.03, 0.1, 2.0, 0.01, 2),
        "MEDIUM": (3000, 0.3, 0.3, 0.02, 0.1, 4.0, 0.01, 3),
        "HEAVY": (8000, 0.5, 0.2, 0.01, 0.05, 8.0, 0.005, 4),
    }
    vals = presets.get(self.hydraulic_preset)
    if vals:
        self.hydraulic_droplets = vals[0]
        self.hydraulic_erosion_rate = vals[1]
        self.hydraulic_deposition_rate = vals[2]
        self.hydraulic_evaporation = vals[3]
        self.hydraulic_inertia = vals[4]
        self.hydraulic_capacity_mult = vals[5]
        self.hydraulic_min_slope = vals[6]
        self.hydraulic_radius = vals[7]


class TILEFORGE_PG_OutdoorSettings(PropertyGroup):
    """Settings for outdoor terrain generation."""

    terrain_type: EnumProperty(
        name="Terrain Type",
        items=TERRAIN_TYPES,
        default="ROCKY",
    )
    noise_type: EnumProperty(
        name="Noise Type",
        items=NOISE_TYPES,
        default="PERLIN",
    )
    noise_basis: EnumProperty(
        name="Noise Basis",
        description="Underlying noise pattern used by the fractal functions",
        items=NOISE_BASES,
        default="PERLIN_ORIGINAL",
    )
    heightmap_image: StringProperty(
        name="Heightmap Image",
        description="Path to grayscale heightmap image",
        subtype='FILE_PATH',
    )

    # Terrain shape controls
    terrain_height_min: FloatProperty(
        name="Min Relief (m)",
        description="Minimum terrain elevation above base in world meters",
        default=0.03,
        min=0.0,
        max=1.0,
        step=1,
        precision=2,
    )
    terrain_height_max: FloatProperty(
        name="Max Relief (m)",
        description="Maximum terrain elevation above base in world meters",
        default=0.5,
        min=0.05,
        max=2.0,
        step=5,
        precision=2,
    )
    noise_scale: FloatProperty(
        name="Noise Scale",
        description="Scale of procedural noise (larger = broader features)",
        default=2.0,
        min=0.1,
        max=20.0,
        step=50,
        precision=2,
    )
    noise_detail: IntProperty(
        name="Noise Detail (Octaves)",
        description="Number of noise octaves for detail",
        default=4,
        min=1,
        max=10,
    )
    noise_roughness: FloatProperty(
        name="Roughness",
        description="How rough/jagged the terrain is",
        default=0.5,
        min=0.0,
        max=1.0,
        step=5,
        precision=2,
    )
    lacunarity: FloatProperty(
        name="Lacunarity",
        description="Frequency multiplier between octaves (higher = more fine detail)",
        default=2.0,
        min=1.0,
        max=4.0,
        step=10,
        precision=2,
    )
    fractal_offset: FloatProperty(
        name="Offset",
        description="Fractal offset — controls base elevation for ridged/hybrid/hetero noise",
        default=1.0,
        min=0.0,
        max=3.0,
        step=10,
        precision=2,
    )
    fractal_gain: FloatProperty(
        name="Gain",
        description="Fractal gain — controls ridge sharpness for ridged/hybrid noise",
        default=2.0,
        min=0.5,
        max=6.0,
        step=10,
        precision=2,
    )
    height_exponent: FloatProperty(
        name="Peak Sharpness",
        description="Height redistribution exponent (>1 = sharp peaks, <1 = flat plateaus)",
        default=1.0,
        min=0.2,
        max=5.0,
        step=10,
        precision=2,
    )
    noise_seed: IntProperty(
        name="Seed",
        description="Random seed for procedural generation",
        default=42,
        min=0,
    )
    subdivisions: IntProperty(
        name="Mesh Subdivisions",
        description="Resolution of the terrain mesh (higher = more detail, slower)",
        default=64,
        min=8,
        max=256,
    )

    # Domain warping
    enable_domain_warp: BoolProperty(
        name="Domain Warping",
        description="Warp noise coordinates for more organic, less grid-aligned terrain",
        default=False,
    )
    warp_strength: FloatProperty(
        name="Warp Strength",
        description="How strongly the noise domain is distorted",
        default=0.3,
        min=0.0,
        max=2.0,
        step=5,
        precision=2,
    )
    warp_scale: FloatProperty(
        name="Warp Scale",
        description="Scale of the warping noise (larger = broader distortions)",
        default=1.5,
        min=0.1,
        max=10.0,
        step=10,
        precision=2,
    )

    # Terracing
    enable_terracing: BoolProperty(
        name="Terracing",
        description="Quantize terrain height into discrete terrace levels",
        default=False,
    )
    terrace_levels: IntProperty(
        name="Terrace Levels",
        description="Number of discrete height levels",
        default=5,
        min=2,
        max=20,
    )
    terrace_sharpness: FloatProperty(
        name="Terrace Sharpness",
        description="Blend between smooth (0) and hard-stepped (1) terraces",
        default=0.7,
        min=0.0,
        max=1.0,
        step=5,
        precision=2,
    )

    # Slope clamping
    enable_slope_clamp: BoolProperty(
        name="Slope Clamping",
        description="Limit maximum slope angle for 3D-printability",
        default=False,
    )
    max_slope_angle: FloatProperty(
        name="Max Slope Angle (deg)",
        description="Maximum allowed slope angle in degrees",
        default=55.0,
        min=15.0,
        max=85.0,
        step=100,
        precision=1,
    )

    # Hydraulic erosion
    enable_hydraulic_erosion: BoolProperty(
        name="Hydraulic Erosion",
        description="Simulate water erosion creating channels and valleys",
        default=False,
    )
    hydraulic_preset: EnumProperty(
        name="Preset",
        items=HYDRAULIC_PRESETS,
        default="MEDIUM",
        update=_update_hydraulic_preset,
    )
    hydraulic_show_advanced: BoolProperty(
        name="Advanced",
        description="Show advanced hydraulic erosion parameters",
        default=False,
    )
    hydraulic_droplets: IntProperty(
        name="Droplets",
        description="Number of water droplets to simulate",
        default=3000,
        min=100,
        max=20000,
    )
    hydraulic_erosion_rate: FloatProperty(
        name="Erosion Rate",
        description="How aggressively water erodes terrain",
        default=0.3,
        min=0.01,
        max=1.0,
        step=5,
        precision=2,
    )
    hydraulic_deposition_rate: FloatProperty(
        name="Deposition Rate",
        description="How quickly sediment is deposited",
        default=0.3,
        min=0.01,
        max=1.0,
        step=5,
        precision=2,
    )
    hydraulic_evaporation: FloatProperty(
        name="Evaporation",
        description="Water evaporation rate per step",
        default=0.02,
        min=0.001,
        max=0.1,
        step=1,
        precision=3,
    )
    hydraulic_inertia: FloatProperty(
        name="Inertia",
        description="How much the droplet resists changing direction",
        default=0.1,
        min=0.0,
        max=1.0,
        step=5,
        precision=2,
    )
    hydraulic_capacity_mult: FloatProperty(
        name="Capacity Multiplier",
        description="Sediment carrying capacity multiplier",
        default=4.0,
        min=0.5,
        max=20.0,
        step=50,
        precision=1,
    )
    hydraulic_min_slope: FloatProperty(
        name="Min Slope",
        description="Minimum slope for capacity calculation",
        default=0.01,
        min=0.001,
        max=0.1,
        step=1,
        precision=3,
    )
    hydraulic_radius: IntProperty(
        name="Erosion Radius",
        description="Radius of erosion brush in grid cells",
        default=3,
        min=1,
        max=8,
    )

    # Thermal erosion
    enable_thermal_erosion: BoolProperty(
        name="Thermal Erosion",
        description="Simulate thermal weathering — material crumbles from steep slopes",
        default=False,
    )
    thermal_talus_angle: FloatProperty(
        name="Talus Angle (deg)",
        description="Angle of repose — slopes steeper than this shed material",
        default=40.0,
        min=10.0,
        max=80.0,
        step=100,
        precision=1,
    )
    thermal_iterations: IntProperty(
        name="Iterations",
        description="Number of thermal erosion passes",
        default=50,
        min=1,
        max=500,
    )
    thermal_strength: FloatProperty(
        name="Strength",
        description="How much material moves per iteration",
        default=0.5,
        min=0.05,
        max=1.0,
        step=5,
        precision=2,
    )

    # Noise layers
    enable_noise_layers: BoolProperty(
        name="Noise Layers",
        description="Add additional noise layers on top of base terrain",
        default=False,
    )
    noise_layers: CollectionProperty(
        type=TILEFORGE_PG_NoiseLayer,
    )
    active_noise_layer_index: IntProperty(
        name="Active Layer",
        default=0,
    )

    # Terrain features
    add_river: BoolProperty(
        name="Add River Channel",
        description="Cut a river channel across the terrain",
        default=False,
    )
    river_curve: PointerProperty(
        name="River Curve",
        description="Curve object defining the river centerline",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'CURVE',
    )
    river_width: FloatProperty(
        name="River Width (m)",
        description="Width of the river channel in world meters",
        default=1.0,
        min=0.3,
        max=3.0,
        step=10,
        precision=1,
    )
    river_depth: FloatProperty(
        name="River Depth (m)",
        description="How deep the river channel is cut in world meters",
        default=0.2,
        min=0.05,
        max=0.6,
        step=5,
        precision=2,
    )
    river_meander_strength: FloatProperty(
        name="Meander Strength",
        description="Noise wobble perpendicular to the curve for organic edges",
        default=0.0,
        min=0.0,
        max=1.0,
        step=5,
        precision=2,
    )

    add_path: BoolProperty(
        name="Add Path / Road",
        description="Flatten a path across the terrain",
        default=False,
    )
    path_curve: PointerProperty(
        name="Path Curve",
        description="Curve object defining the path/road centerline",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'CURVE',
    )
    path_width: FloatProperty(
        name="Path Width (m)",
        description="Width of the flattened path in world meters",
        default=1.5,
        min=0.3,
        max=3.0,
        step=10,
        precision=1,
    )

    floor_texture: EnumProperty(
        name="Ground Texture",
        items=FLOOR_TEXTURES,
        default="DIRT",
    )
    texture_strength: FloatProperty(
        name="Texture Strength",
        description="Intensity of surface texture displacement",
        default=0.3,
        min=0.0,
        max=1.0,
        step=5,
        precision=2,
    )


class TILEFORGE_PG_DungeonSettings(PropertyGroup):
    """Settings for dungeon / interior generation."""

    dungeon_source: EnumProperty(
        name="Source",
        items=DUNGEON_SOURCE,
        default="IMAGE",
    )
    floorplan_image: StringProperty(
        name="Floor Plan Image",
        description="Path to black/white floor plan image (white=floor, black=wall)",
        subtype='FILE_PATH',
    )
    wall_height: FloatProperty(
        name="Wall Height (m)",
        description="Height of dungeon walls in world meters",
        default=1.8,
        min=0.6,
        max=5.0,
        step=10,
        precision=1,
    )
    wall_thickness: FloatProperty(
        name="Wall Thickness (m)",
        description="Thickness of dungeon walls in world meters",
        default=0.3,
        min=0.1,
        max=1.0,
        step=5,
        precision=1,
    )
    doorway_width: FloatProperty(
        name="Doorway Width (m)",
        description="Width of auto-detected doorway openings in world meters",
        default=1.5,
        min=1.0,
        max=3.0,
        step=10,
        precision=1,
    )
    doorway_height: FloatProperty(
        name="Doorway Height (m)",
        description="Height of doorway openings in world meters (0 = full wall height)",
        default=1.5,
        min=0.0,
        max=4.0,
        step=10,
        precision=1,
    )

    floor_texture: EnumProperty(
        name="Floor Texture",
        items=FLOOR_TEXTURES,
        default="STONE_SLAB",
    )
    texture_strength: FloatProperty(
        name="Texture Strength",
        description="Intensity of floor texture displacement",
        default=0.2,
        min=0.0,
        max=1.0,
        step=5,
        precision=2,
    )

    # Elevation / stairs
    add_stairs: BoolProperty(
        name="Include Stairs",
        description="Add staircase sections for elevation changes",
        default=False,
    )
    stair_step_height: FloatProperty(
        name="Step Height (m)",
        description="Height of each stair step in world meters",
        default=0.2,
        min=0.05,
        max=0.4,
        step=1,
        precision=2,
    )
    stair_step_depth: FloatProperty(
        name="Step Depth (m)",
        description="Depth (run) of each stair step in world meters",
        default=0.4,
        min=0.2,
        max=0.8,
        step=5,
        precision=1,
    )


class TILEFORGE_PG_ExportSettings(PropertyGroup):
    """Settings for STL export."""

    export_path: StringProperty(
        name="Export Folder",
        description="Folder to export STL files into",
        subtype='DIR_PATH',
        default="//tile_export/",
    )
    filename_prefix: StringProperty(
        name="Filename Prefix",
        description="Prefix for exported tile filenames",
        default="tile",
    )
    check_manifold: BoolProperty(
        name="Manifold Check",
        description="Validate mesh is manifold (watertight) before export",
        default=True,
    )
    check_thickness: BoolProperty(
        name="Wall Thickness Check",
        description="Warn if any walls are thinner than minimum",
        default=True,
    )
    min_wall_thickness: FloatProperty(
        name="Min Wall Thickness (mm)",
        description="Minimum acceptable wall thickness for printing",
        default=1.2,
        min=0.4,
        max=3.0,
        step=10,
        precision=2,
    )


class TILEFORGE_PG_Main(PropertyGroup):
    """Top-level property group holding all sub-groups."""

    generation_mode: EnumProperty(
        name="Mode",
        items=GENERATION_MODES,
        default="OUTDOOR",
    )

    tile: PointerProperty(type=TILEFORGE_PG_TileSettings)
    outdoor: PointerProperty(type=TILEFORGE_PG_OutdoorSettings)
    dungeon: PointerProperty(type=TILEFORGE_PG_DungeonSettings)
    export: PointerProperty(type=TILEFORGE_PG_ExportSettings)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = (
    TILEFORGE_PG_TileSettings,
    TILEFORGE_PG_NoiseLayer,
    TILEFORGE_PG_OutdoorSettings,
    TILEFORGE_PG_DungeonSettings,
    TILEFORGE_PG_ExportSettings,
    TILEFORGE_PG_Main,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.tile_forge = PointerProperty(type=TILEFORGE_PG_Main)


def unregister():
    del bpy.types.Scene.tile_forge
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
