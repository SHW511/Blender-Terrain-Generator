# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

D&D Tile Forge is a Blender 5.0+ extension that generates 3D-printable modular terrain tiles for tabletop RPGs. It supports outdoor procedural terrain (noise-based or heightmap) and indoor dungeon generation from floor plan images. Tiles are sliced into modular pieces with interlocking connectors and exported as STL files.

## Development Setup

This is a Blender extension — there is no standalone build, test, or lint pipeline. The code runs inside Blender's embedded Python interpreter.

- **Install**: Place the folder in Blender's `extensions/user_default/` directory and enable via Edit > Preferences > Extensions
- **Reload after changes**: In Blender, disable and re-enable the extension, or restart Blender
- **No external dependencies**: Only uses Blender's built-in Python modules (`bpy`, `bmesh`, `mathutils`, `bpy_extras`)

## Architecture

```
__init__.py          → register/unregister entry point, imports all modules
properties.py        → PropertyGroup definitions (all user-facing settings)
operators.py         → Blender operators (7 total: generate, slice, export, cleanup)
ui.py                → UI panels in the 3D Viewport sidebar (category: "Tile Forge")
mesh_gen.py          → Core mesh generation: terrain, dungeons, connectors, slicing, export
```

**Data flow**: UI panels (`ui.py`) display properties (`properties.py`). Operators (`operators.py`) read those properties and call mesh generation functions (`mesh_gen.py`).

### Naming Convention

All Blender-registered classes use the `TILEFORGE_` prefix:
- Operators: `TILEFORGE_OT_*`
- Panels: `TILEFORGE_PT_*`
- Property Groups: `TILEFORGE_PG_*`

### Key Property Groups (properties.py)

- `TILEFORGE_PG_Main` — top-level container attached to `bpy.types.Scene`
- `TILEFORGE_PG_TileSettings` — tile dimensions, grid, connector type
- `TILEFORGE_PG_OutdoorSettings` — terrain type, noise params, heightmap
- `TILEFORGE_PG_DungeonSettings` — wall height, doorways, stairs
- `TILEFORGE_PG_ExportSettings` — export folder, manifold checks

### Generation Modes

- **OUTDOOR**: Procedural noise (Perlin/Voronoi/Musgrave) or custom heightmap → full terrain → slice into tiles
- **DUNGEON**: Black/white floor plan image (white=floor, black=wall) → 3D structure → slice into tiles

## Units & Coordinate System

- Internal calculations use Blender units (meters)
- `mm()` helper in `mesh_gen.py` converts millimeters to Blender units (divides by 1000)
- STL export applies 1000x scale factor to convert back to millimeters for 3D printing slicers
- Grid square size depends on tile size, print scale, and `grid_squares` setting (default 4 squares across a 6m tile at 1:60 = 25mm)
- Grid line engravings are 0.4–0.6mm deep

## Key Technical Details

- Uses `bmesh` for all low-level mesh construction and manipulation
- Boolean modifier operations for grid engraving, connectors, and tile slicing
- Interlocking connector system: pegs on east/south edges, slots on west/north edges
- `check_manifold()` validates watertight meshes before export
- Generated objects use `TileForge_` prefix in Blender's scene for cleanup tracking
