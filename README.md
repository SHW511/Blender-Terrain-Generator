# D&D Tile Forge

A Blender 5.0+ extension that generates 3D-printable modular terrain tiles for tabletop RPGs. Create outdoor landscapes or dungeon interiors, slice them into interlocking tiles, and export as STL files ready for your 3D printer.

![Blender](https://img.shields.io/badge/Blender-5.0%2B-orange)
![License](https://img.shields.io/badge/License-GPL--3.0--or--later-blue)

## Features

### Outdoor Terrain
- **Procedural noise** -- Perlin, Voronoi, Ridged Multi-Fractal, and more with configurable octaves, lacunarity, and seed
- **Terrain presets** -- Flat/Grassland, Rocky/Mountain, Desert Dunes, Forest Floor
- **Custom heightmap** -- Import a grayscale image as elevation data
- **Color map mode** -- Derive terrain heights from an illustrated color map with configurable color zones
- **AI heightmap generation** -- Generate heightmaps from a source image using OpenAI (gpt-image-1) or Google Gemini
- **Noise layers** -- Stack multiple noise layers with blend modes (Add, Multiply, Screen, Overlay)
- **Terrain shaping** -- Domain warping, terracing, and slope clamping
- **Erosion simulation** -- Hydraulic erosion (with presets) and thermal erosion
- **Curve-based features** -- Rivers, paths, ridges, and cliffs driven by Blender curves
- **Procedural road network** -- A* pathfinding between waypoints with slope avoidance
- **Paint tools** -- Paint heightmaps and road masks directly onto the terrain

### Dungeon / Interior
- **Floor plan import** -- Load a black & white image (white = floor, black = wall) to generate 3D dungeon geometry
- **Configurable wall height** for different room styles
- **Floor textures** -- Stone Slab, Cobblestone, Wood Plank, Dirt

### Tile System
- **Automatic slicing** -- Cut terrain or dungeons into a grid of modular tiles
- **Interlocking connectors** -- Peg & Slot or Tongue & Groove for easy assembly
- **Configurable tolerances** for tuning fit on your specific printer
- **Grid line engraving** -- 1-inch, 25mm, or custom grid squares engraved into tile surfaces
- **Print dimension readout** -- Live display of tile, grid square, and total map size in mm

### Export
- **STL export** with automatic mm scaling for slicer compatibility
- **Manifold checking** -- Validate watertight meshes before export
- **Wall thickness checking** -- Catch thin features that may fail to print
- **Batch or single tile export**

## Installation

### Recommended: Add as a Remote Repository (auto-updates)

This is the easiest method and will keep the extension up to date automatically.

1. Open Blender 5.0 or later
2. Go to **Edit > Preferences > Get Extensions**
3. Open the dropdown in the top-right corner (next to the **Repositories** header) and click **Add Remote Repository**
4. Enter the following URL:
   ```
   https://github.com/SHW511/Blender-Terrain-Generator.git
   ```
5. Click **Create**
6. Search for **D&D Tile Forge** in the extensions list and install it

The extension will now appear in the **Get Extensions** tab and receive updates through Blender's built-in update system.

### Alternative: Manual Install

1. Download the latest `.zip` from the [Releases](https://github.com/SHW511/Blender-Terrain-Generator/releases) page
2. In Blender, go to **Edit > Preferences > Get Extensions**
3. Open the dropdown in the top-right corner and click **Install from Disk...**
4. Select the downloaded `.zip` file

## Usage

1. Open the **Tile Forge** tab in the 3D Viewport sidebar (press `N` to toggle the sidebar)
2. Choose a generation mode: **Outdoor Terrain** or **Dungeon / Interior**
3. Configure your settings (tile size, terrain type, grid squares, etc.)
4. Click **Generate Terrain** or **Generate Dungeon**
5. Adjust the result, then use **Slice into Tiles** to cut it into modular pieces
6. **Export All Tiles** to write STL files for 3D printing

### AI Heightmap Generation

To use AI-powered heightmap generation:

1. Go to **Edit > Preferences > Extensions > D&D Tile Forge**
2. Enter your API key for OpenAI or Google Gemini
3. In the **Heightmap Tools** sub-panel, select a source image and provider
4. Click **Generate Heightmap**

### Paint Tools

- **Paint Heightmap** -- Paint elevation zones (Sea, Low, Mid, High, Peak) onto a reference image to create custom heightmaps
- **Paint Roads** -- Paint road masks directly onto generated terrain, with optional cobblestone/flagstone textures

## Requirements

- **Blender 5.0** or later
- No external Python dependencies -- uses only Blender's built-in modules
- Network access is only required for the optional AI heightmap feature

## License

GPL-3.0-or-later
