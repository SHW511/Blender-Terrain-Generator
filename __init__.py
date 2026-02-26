"""
D&D Tile Forge - Blender Extension
Generate 3D-printable modular terrain tiles for tabletop RPGs.

Supports outdoor terrain (heightmap/procedural) and dungeon/interior maps
with configurable tile sizes, interlocking connectors, and batch STL export.
"""

from . import ai_api
from . import properties
from . import operators
from . import ui


def register():
    properties.register()
    operators.register()
    ui.register()


def unregister():
    ui.unregister()
    operators.unregister()
    properties.unregister()
