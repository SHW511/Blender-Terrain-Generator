"""
Core mesh generation utilities for D&D Tile Forge.
Provides functions for creating base grids, applying noise, heightmaps,
connectors, grid line engraving, and dungeon wall extrusion.

Two unit systems:
- World dimensions (m): terrain size, tile size, heights — converted via m()
- Physical dimensions (mm): grid lines, connectors, base height — converted via mm()

Both convert to Blender units (meters internally).
"""

import colorsys
import math
import bmesh
import bpy
from mathutils import Vector, noise as mathnoise


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def mm(value):
    """Convert millimeters to Blender units (meters)."""
    return value * 0.001


def m(value, print_scale):
    """Convert world meters to Blender units at the given print scale.

    At D&D standard 1:60 scale, 1 world meter = 16.67mm physical.
    """
    return value / print_scale


# ---------------------------------------------------------------------------
# Base grid creation
# ---------------------------------------------------------------------------

def create_base_grid(name, size_x, size_y, subdivisions, print_scale):
    """
    Create a subdivided plane mesh centered at origin.

    Args:
        name: Object name
        size_x: Width in world meters
        size_y: Depth in world meters
        subdivisions: Number of cuts along each axis
        print_scale: World-to-print scale ratio

    Returns:
        (object, mesh) tuple
    """
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)

    bm = bmesh.new()
    bmesh.ops.create_grid(
        bm,
        x_segments=subdivisions,
        y_segments=subdivisions,
        size=1.0,  # Will be scaled
    )

    # Scale to target dimensions
    sx = m(size_x, print_scale) / 2.0
    sy = m(size_y, print_scale) / 2.0
    for v in bm.verts:
        v.co.x *= sx
        v.co.y *= sy

    bm.to_mesh(mesh)
    bm.free()

    return obj, mesh


# ---------------------------------------------------------------------------
# Procedural noise terrain
# ---------------------------------------------------------------------------

# Terrain presets: multipliers and recommended defaults per terrain type
TERRAIN_PRESETS = {
    "FLAT": {
        "scale_mult": 0.5, "height_mult": 0.3, "roughness": 0.3,
        "noise_type": "HYBRID", "noise_basis": "PERLIN_NEW",
        "exponent": 0.6, "lacunarity": 1.8,
        "domain_warp": False,
    },
    "ROCKY": {
        "scale_mult": 1.0, "height_mult": 1.0, "roughness": 0.7,
        "noise_type": "RIDGED", "noise_basis": "PERLIN_ORIGINAL",
        "exponent": 1.3, "lacunarity": 2.0,
        "domain_warp": True, "warp_strength": 0.3,
    },
    "DESERT": {
        "scale_mult": 2.0, "height_mult": 0.6, "roughness": 0.2,
        "noise_type": "HYBRID", "noise_basis": "PERLIN_NEW",
        "exponent": 0.8, "lacunarity": 1.5,
        "domain_warp": True, "warp_strength": 0.5,
    },
    "FOREST": {
        "scale_mult": 1.5, "height_mult": 0.5, "roughness": 0.55,
        "noise_type": "HETERO_TERRAIN", "noise_basis": "PERLIN_NEW",
        "exponent": 0.9, "lacunarity": 2.2,
        "domain_warp": True, "warp_strength": 0.4,
    },
}


def _evaluate_noise(sample, noise_type, basis, H, lacunarity, octaves, offset, gain):
    """Evaluate a single noise sample. Returns float in roughly -1..1 range.

    Args:
        sample: Vector position in noise space
        noise_type: One of NOISE_TYPES identifiers
        basis: Noise basis string
        H: Fractal dimension parameter
        lacunarity: Frequency multiplier between octaves
        octaves: Number of noise octaves
        offset: Fractal offset (for ridged/hybrid/hetero)
        gain: Fractal gain (for ridged/hybrid)

    Returns:
        Noise value roughly in -1..1 range
    """
    if noise_type == "PERLIN":
        n = mathnoise.fractal(
            sample, H, lacunarity, octaves,
            noise_basis=basis,
        )
    elif noise_type == "VORONOI":
        cell = mathnoise.voronoi(sample, distance_metric='DISTANCE')
        n = cell[0][0] if cell else 0.0
        n = (0.5 - n) * 2.0
    elif noise_type == "MULTI_FRACTAL":
        n = mathnoise.multi_fractal(
            sample, H, lacunarity, octaves,
            noise_basis=basis,
        )
        n = n - 1.0
    elif noise_type == "HETERO_TERRAIN":
        n = mathnoise.hetero_terrain(
            sample, H, lacunarity, octaves, offset,
            noise_basis=basis,
        )
        n = (n - offset) * 0.5
    elif noise_type == "RIDGED":
        n = mathnoise.ridged_multi_fractal(
            sample, H, lacunarity, octaves, offset, gain,
            noise_basis=basis,
        )
        n = (n - 1.0) / max(gain * 0.5, 0.5)
    elif noise_type == "HYBRID":
        n = mathnoise.hybrid_multi_fractal(
            sample, H, lacunarity, octaves, offset, gain,
            noise_basis=basis,
        )
        n = (n - offset) * 0.5
    else:
        n = mathnoise.fractal(
            sample, H, lacunarity, octaves,
            noise_basis=basis,
        )
    return n


def apply_procedural_noise(obj, settings_outdoor, settings_tile):
    """
    Displace vertices of obj using procedural noise.

    Args:
        obj: Blender mesh object
        settings_outdoor: TILEFORGE_PG_OutdoorSettings
        settings_tile: TILEFORGE_PG_TileSettings
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    print_scale = settings_tile.print_scale
    terrain_type = settings_outdoor.terrain_type
    preset = TERRAIN_PRESETS.get(terrain_type, TERRAIN_PRESETS["ROCKY"])

    noise_scale = settings_outdoor.noise_scale * preset["scale_mult"]
    height_range = m(settings_outdoor.terrain_height_max - settings_outdoor.terrain_height_min, print_scale)
    height_offset = m(settings_outdoor.terrain_height_min, print_scale)
    seed = settings_outdoor.noise_seed
    octaves = settings_outdoor.noise_detail
    H = 1.0 - settings_outdoor.noise_roughness * preset.get("roughness", 1.0)
    lacunarity = settings_outdoor.lacunarity * preset.get("lacunarity", 2.0) / 2.0
    basis = preset.get("noise_basis", settings_outdoor.noise_basis)
    offset = settings_outdoor.fractal_offset
    gain = settings_outdoor.fractal_gain
    exponent = settings_outdoor.height_exponent * preset.get("exponent", 1.0)
    noise_type = preset.get("noise_type", settings_outdoor.noise_type)

    # Compute mesh extents for normalization
    half_x = m(settings_tile.map_width, print_scale) / 2.0
    half_y = m(settings_tile.map_depth, print_scale) / 2.0

    # Seed offset in noise space (small, varied)
    seed_offset = Vector(((seed * 13.37) % 1000.0, (seed * 7.42) % 1000.0, 0.0))

    # Domain warping and terracing parameters
    domain_warp_enabled = settings_outdoor.enable_domain_warp
    warp_strength_val = settings_outdoor.warp_strength
    warp_scale_val = settings_outdoor.warp_scale
    terrace_enabled = settings_outdoor.enable_terracing
    terrace_levels_val = settings_outdoor.terrace_levels
    terrace_sharpness_val = settings_outdoor.terrace_sharpness

    for v in bm.verts:
        # Normalize position to 0..1 across mesh, then scale to noise space
        nx = (v.co.x + half_x) / (2.0 * half_x) if half_x > 0 else 0.5
        ny = (v.co.y + half_y) / (2.0 * half_y) if half_y > 0 else 0.5
        sample = Vector((nx * noise_scale, ny * noise_scale, 0.0)) + seed_offset

        # Domain warping: distort sample coordinates for organic, non-grid patterns
        if domain_warp_enabled:
            warp_sample = Vector((nx * warp_scale_val, ny * warp_scale_val, 0.0)) + seed_offset
            warp_vec = mathnoise.turbulence_vector(warp_sample, 3, False)
            sample = sample + Vector((warp_vec.x, warp_vec.y, 0.0)) * warp_strength_val

        n = _evaluate_noise(sample, noise_type, basis, H, lacunarity, octaves, offset, gain)

        # Map noise (roughly -1..1) to height range
        normalized = (n + 1.0) / 2.0  # 0..1
        normalized = max(0.0, min(1.0, normalized))

        # Apply height redistribution curve
        if exponent != 1.0:
            normalized = pow(normalized, exponent)

        # Terracing: quantize height to discrete levels, blend with smooth
        if terrace_enabled and terrace_levels_val > 1:
            terraced = round(normalized * terrace_levels_val) / terrace_levels_val
            normalized = normalized + (terraced - normalized) * terrace_sharpness_val

        height = height_offset + normalized * height_range * preset["height_mult"]

        v.co.z = height

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


# ---------------------------------------------------------------------------
# Slope clamping (thermal-erosion-inspired)
# ---------------------------------------------------------------------------

def clamp_slopes(obj, max_angle_deg, iterations=5):
    """
    Iteratively smooth the mesh so no slope exceeds max_angle_deg.
    Uses a thermal-erosion-inspired algorithm: excess height is redistributed
    from higher to lower vertices along each edge.

    Args:
        obj: Blender mesh object
        max_angle_deg: Maximum allowed slope in degrees
        iterations: Number of smoothing passes (default 5)
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    max_tan = math.tan(math.radians(max_angle_deg))

    for _ in range(iterations):
        adjustments = [0.0] * len(bm.verts)
        link_counts = [0] * len(bm.verts)
        any_exceeded = False

        for edge in bm.edges:
            v0, v1 = edge.verts
            dx = v1.co.x - v0.co.x
            dy = v1.co.y - v0.co.y
            horiz_dist = math.sqrt(dx * dx + dy * dy)
            if horiz_dist < 1e-10:
                continue

            dz = v1.co.z - v0.co.z
            slope = abs(dz) / horiz_dist

            if slope > max_tan:
                any_exceeded = True
                max_dz = max_tan * horiz_dist
                excess = abs(dz) - max_dz

                # Transfer half of excess from higher to lower vertex
                transfer = excess * 0.5
                if dz > 0:
                    # v1 is higher
                    adjustments[v1.index] -= transfer
                    adjustments[v0.index] += transfer
                else:
                    # v0 is higher
                    adjustments[v0.index] -= transfer
                    adjustments[v1.index] += transfer

                link_counts[v0.index] += 1
                link_counts[v1.index] += 1

        if not any_exceeded:
            break

        # Apply adjustments — both sides divided by their own link count
        # to keep the correction symmetric and conserve total height
        for v in bm.verts:
            if link_counts[v.index] > 0:
                v.co.z += adjustments[v.index] / max(link_counts[v.index], 1)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


# ---------------------------------------------------------------------------
# Height grid infrastructure (for erosion algorithms)
# ---------------------------------------------------------------------------

def _build_grid_index_map(bm, grid_size):
    """Build position-based (row, col) -> vert_index map.

    Sorts vertices by Y then X to determine grid position, which is robust
    regardless of bmesh vertex ordering.

    Args:
        bm: BMesh with a grid of grid_size x grid_size vertices
        grid_size: Number of vertices along each axis

    Returns:
        dict mapping (row, col) tuples to vertex indices
    """
    bm.verts.ensure_lookup_table()
    # Sort by Y (row) then X (col) to determine grid order
    sorted_verts = sorted(bm.verts, key=lambda v: (v.co.y, v.co.x))
    index_map = {}
    for i, v in enumerate(sorted_verts):
        row = i // grid_size
        col = i % grid_size
        index_map[(row, col)] = v.index
    return index_map


def mesh_to_height_grid(bm, grid_size):
    """Extract vertex Z heights into a 2D list [row][col].

    Args:
        bm: BMesh with grid_size x grid_size vertices
        grid_size: Number of vertices per axis (subdivisions + 1)

    Returns:
        (grid, index_map) where grid is a 2D list of floats and
        index_map maps (row, col) -> vertex index
    """
    index_map = _build_grid_index_map(bm, grid_size)
    bm.verts.ensure_lookup_table()
    grid = []
    for row in range(grid_size):
        row_data = []
        for col in range(grid_size):
            vi = index_map[(row, col)]
            row_data.append(bm.verts[vi].co.z)
        grid.append(row_data)
    return grid, index_map


def height_grid_to_mesh(bm, grid, grid_size, index_map):
    """Write 2D height list back to bmesh vertices.

    Args:
        bm: BMesh to update
        grid: 2D list [row][col] of Z heights
        grid_size: Number of vertices per axis
        index_map: Mapping from (row, col) -> vertex index
    """
    bm.verts.ensure_lookup_table()
    for row in range(grid_size):
        for col in range(grid_size):
            vi = index_map[(row, col)]
            bm.verts[vi].co.z = grid[row][col]


# ---------------------------------------------------------------------------
# Hydraulic erosion
# ---------------------------------------------------------------------------

def _bilinear_height(grid, grid_size, px, py):
    """Interpolated height at float position (px, py) on the grid."""
    x0 = max(0, int(px))
    y0 = max(0, int(py))
    x1 = min(x0 + 1, grid_size - 1)
    y1 = min(y0 + 1, grid_size - 1)

    fx = px - x0
    fy = py - y0

    h00 = grid[y0][x0]
    h10 = grid[y0][x1]
    h01 = grid[y1][x0]
    h11 = grid[y1][x1]

    return (h00 * (1 - fx) * (1 - fy) +
            h10 * fx * (1 - fy) +
            h01 * (1 - fx) * fy +
            h11 * fx * fy)


def _bilinear_gradient(grid, grid_size, px, py):
    """Compute (gx, gy) gradient at float position via bilinear interpolation."""
    x0 = max(0, int(px))
    y0 = max(0, int(py))
    x1 = min(x0 + 1, grid_size - 1)
    y1 = min(y0 + 1, grid_size - 1)

    fx = px - x0
    fy = py - y0

    h00 = grid[y0][x0]
    h10 = grid[y0][x1]
    h01 = grid[y1][x0]
    h11 = grid[y1][x1]

    gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
    gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx

    return gx, gy


def _erode_at_point(grid, grid_size, px, py, amount, radius):
    """Distribute erosion in a weighted circular brush around (px, py)."""
    x0 = int(px)
    y0 = int(py)
    total_weight = 0.0
    weights = []

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx = x0 + dx
            ny = y0 + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                dist_sq = (nx - px) ** 2 + (ny - py) ** 2
                r_sq = radius * radius
                if dist_sq < r_sq:
                    w = max(0.0, 1.0 - dist_sq / r_sq)
                    weights.append((ny, nx, w))
                    total_weight += w

    if total_weight > 0:
        for ry, rx, w in weights:
            grid[ry][rx] -= amount * (w / total_weight)


def _deposit_at_point(grid, grid_size, px, py, amount):
    """Bilinear deposition of sediment at float position."""
    x0 = max(0, int(px))
    y0 = max(0, int(py))
    x1 = min(x0 + 1, grid_size - 1)
    y1 = min(y0 + 1, grid_size - 1)

    fx = px - x0
    fy = py - y0

    grid[y0][x0] += amount * (1 - fx) * (1 - fy)
    grid[y0][x1] += amount * fx * (1 - fy)
    grid[y1][x0] += amount * (1 - fx) * fy
    grid[y1][x1] += amount * fx * fy


def apply_hydraulic_erosion(grid, grid_size, settings_outdoor, cell_size):
    """Particle-based droplet erosion simulation. Modifies grid in-place.

    Args:
        grid: 2D list [row][col] of Z heights
        grid_size: Number of vertices per axis
        settings_outdoor: TILEFORGE_PG_OutdoorSettings
        cell_size: Distance between adjacent grid cells in Blender units
    """
    import random

    num_droplets = settings_outdoor.hydraulic_droplets
    erosion_rate = settings_outdoor.hydraulic_erosion_rate
    deposition_rate = settings_outdoor.hydraulic_deposition_rate
    evaporation = settings_outdoor.hydraulic_evaporation
    inertia = settings_outdoor.hydraulic_inertia
    capacity_mult = settings_outdoor.hydraulic_capacity_mult
    min_slope = settings_outdoor.hydraulic_min_slope
    radius = settings_outdoor.hydraulic_radius
    max_steps = 80

    rng = random.Random(settings_outdoor.noise_seed)

    for _ in range(num_droplets):
        # Random start position within interior
        px = rng.uniform(1, grid_size - 2)
        py = rng.uniform(1, grid_size - 2)
        dir_x = 0.0
        dir_y = 0.0
        velocity = 1.0
        water = 1.0
        sediment = 0.0

        for step in range(max_steps):
            # Current height
            h_old = _bilinear_height(grid, grid_size, px, py)

            # Compute gradient
            gx, gy = _bilinear_gradient(grid, grid_size, px, py)

            # Update direction with inertia
            dir_x = dir_x * inertia - gx * (1.0 - inertia)
            dir_y = dir_y * inertia - gy * (1.0 - inertia)

            # Normalize direction
            length = math.sqrt(dir_x * dir_x + dir_y * dir_y)
            if length < 1e-10:
                # Deposit remaining sediment before stopping
                if sediment > 0:
                    _deposit_at_point(grid, grid_size, px, py, sediment)
                break
            dir_x /= length
            dir_y /= length

            # Move one step
            new_px = px + dir_x
            new_py = py + dir_y

            # Stop if out of bounds — deposit remaining sediment
            if (new_px < 1 or new_px >= grid_size - 1 or
                    new_py < 1 or new_py >= grid_size - 1):
                if sediment > 0:
                    _deposit_at_point(grid, grid_size, px, py, sediment)
                break

            h_new = _bilinear_height(grid, grid_size, new_px, new_py)
            dh = h_new - h_old

            # Going uphill: deposit sediment and stop
            if dh > 0:
                _deposit_at_point(grid, grid_size, px, py, sediment)
                break

            # Going downhill: compute carry capacity
            capacity = max(-dh, min_slope) * velocity * water * capacity_mult

            if sediment > capacity:
                # Deposit excess
                deposit = (sediment - capacity) * deposition_rate
                _deposit_at_point(grid, grid_size, px, py, deposit)
                sediment -= deposit
            else:
                # Erode
                erode_amount = min((capacity - sediment) * erosion_rate, -dh)
                _erode_at_point(grid, grid_size, px, py, erode_amount, radius)
                sediment += erode_amount

            # Update velocity and water
            velocity = math.sqrt(max(velocity * velocity - dh, 0.0))
            water *= (1.0 - evaporation)

            px = new_px
            py = new_py

            if water < 0.001:
                # Deposit remaining sediment before evaporating
                if sediment > 0:
                    _deposit_at_point(grid, grid_size, px, py, sediment)
                break


# ---------------------------------------------------------------------------
# Thermal erosion
# ---------------------------------------------------------------------------

def apply_thermal_erosion(grid, grid_size, settings_outdoor, cell_size):
    """Thermal weathering — material crumbles from steep slopes, accumulates at base.

    Unlike clamp_slopes() which just reduces heights, thermal erosion conserves
    material: height removed from high vertices is added to low vertices,
    creating visible talus fans.

    Args:
        grid: 2D list [row][col] of Z heights
        grid_size: Number of vertices per axis
        settings_outdoor: TILEFORGE_PG_OutdoorSettings
        cell_size: Distance between adjacent grid cells in Blender units
    """
    talus_angle = settings_outdoor.thermal_talus_angle
    iterations = settings_outdoor.thermal_iterations
    strength = settings_outdoor.thermal_strength

    max_dz = math.tan(math.radians(talus_angle)) * cell_size
    max_dz_diag = max_dz * math.sqrt(2.0)

    # 8-neighbor offsets: (drow, dcol, is_diagonal)
    neighbors = [
        (-1, -1, True), (-1, 0, False), (-1, 1, True),
        (0, -1, False),                  (0, 1, False),
        (1, -1, True),  (1, 0, False),  (1, 1, True),
    ]

    for _ in range(iterations):
        # Double-buffer: accumulate deltas
        delta = [[0.0] * grid_size for _ in range(grid_size)]

        for row in range(1, grid_size - 1):
            for col in range(1, grid_size - 1):
                h = grid[row][col]

                # Find neighbors that exceed threshold
                exceeding = []
                for dr, dc, is_diag in neighbors:
                    nr, nc = row + dr, col + dc
                    nh = grid[nr][nc]
                    dh = h - nh
                    threshold = max_dz_diag if is_diag else max_dz
                    if dh > threshold:
                        exceeding.append((nr, nc, dh, threshold))

                if not exceeding:
                    continue

                num = len(exceeding)
                for nr, nc, dh, threshold in exceeding:
                    excess = dh - threshold
                    transfer = strength * excess * 0.5 / num
                    delta[row][col] -= transfer
                    delta[nr][nc] += transfer

        # Apply deltas
        for row in range(grid_size):
            for col in range(grid_size):
                grid[row][col] += delta[row][col]


# ---------------------------------------------------------------------------
# Multi-layer noise composition
# ---------------------------------------------------------------------------

def apply_noise_layers(obj, settings_outdoor, settings_tile):
    """Apply additional noise layers on top of base terrain.

    Args:
        obj: Blender mesh object with terrain heights already applied
        settings_outdoor: TILEFORGE_PG_OutdoorSettings
        settings_tile: TILEFORGE_PG_TileSettings
    """
    layers = settings_outdoor.noise_layers
    if not layers:
        return

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    print_scale = settings_tile.print_scale
    half_x = m(settings_tile.map_width, print_scale) / 2.0
    half_y = m(settings_tile.map_depth, print_scale) / 2.0
    height_min = m(settings_outdoor.terrain_height_min, print_scale)
    height_max = m(settings_outdoor.terrain_height_max, print_scale)
    height_range = height_max - height_min
    base_seed = settings_outdoor.noise_seed

    for layer in layers:
        if not layer.enabled:
            continue

        layer_scale = layer.scale
        layer_strength = m(layer.strength, print_scale)
        layer_seed = base_seed + layer.seed_offset
        seed_vec = Vector(((layer_seed * 13.37) % 1000.0,
                           (layer_seed * 7.42) % 1000.0, 0.0))
        layer_octaves = layer.octaves
        layer_noise_type = layer.noise_type
        layer_basis = layer.noise_basis
        use_mask = layer.use_mask
        mask_invert = layer.mask_invert
        mask_contrast = layer.mask_contrast
        blend_mode = layer.blend_mode

        for v in bm.verts:
            nx = (v.co.x + half_x) / (2.0 * half_x) if half_x > 0 else 0.5
            ny = (v.co.y + half_y) / (2.0 * half_y) if half_y > 0 else 0.5
            sample = Vector((nx * layer_scale, ny * layer_scale, 0.0)) + seed_vec

            n = _evaluate_noise(sample, layer_noise_type, layer_basis,
                                0.5, 2.0, layer_octaves, 1.0, 2.0)
            # Normalize to 0..1
            noise_norm = max(0.0, min(1.0, (n + 1.0) / 2.0))

            # Compute mask
            mask = 1.0
            if use_mask and height_range > 0:
                norm_z = max(0.0, min(1.0,
                             (v.co.z - height_min) / height_range))
                mask = pow(norm_z, mask_contrast)
                if mask_invert:
                    mask = 1.0 - mask

            # Compute layer value
            layer_value = (noise_norm - 0.5) * 2.0 * layer_strength

            # Blend
            if blend_mode == "ADD":
                v.co.z += layer_value * mask
            elif blend_mode == "MULTIPLY":
                strength_factor = layer_strength / max(height_range, 1e-6)
                v.co.z *= (1.0 + (noise_norm - 0.5) * strength_factor * mask)
            elif blend_mode == "SCREEN":
                norm_z = max(0.0, min(1.0,
                             (v.co.z - height_min) / height_range)) if height_range > 0 else 0.5
                v.co.z += layer_value * (1.0 - norm_z) * mask
            elif blend_mode == "OVERLAY":
                norm_z = max(0.0, min(1.0,
                             (v.co.z - height_min) / height_range)) if height_range > 0 else 0.5
                if norm_z < 0.5:
                    factor = 2.0 * norm_z * noise_norm
                else:
                    factor = 1.0 - 2.0 * (1.0 - norm_z) * (1.0 - noise_norm)
                target_z = height_min + factor * height_range
                v.co.z += (target_z - v.co.z) * layer_strength / max(height_range, 1e-6) * mask

            # Clamp
            v.co.z = max(height_min, min(height_max, v.co.z))

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


# ---------------------------------------------------------------------------
# Ground textures (micro-displacement)
# ---------------------------------------------------------------------------

# Maximum texture displacement: 0.5mm (physical)
_MAX_DISP = mm(0.5)


def apply_ground_texture(obj, floor_texture, texture_strength, settings_tile,
                         floor_only_z=None):
    """Apply micro-displacement ground texture to mesh surface.

    Args:
        obj: Blender mesh object
        floor_texture: Texture enum string (STONE_SLAB, COBBLESTONE, etc.)
        texture_strength: 0..1 intensity multiplier
        settings_tile: TILEFORGE_PG_TileSettings
        floor_only_z: If set, skip vertices with z > this value (dungeon walls)
    """
    if texture_strength <= 0.0:
        return

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    print_scale = settings_tile.print_scale
    half_x = m(settings_tile.map_width, print_scale) / 2.0
    half_y = m(settings_tile.map_depth, print_scale) / 2.0

    dispatch = {
        "STONE_SLAB": _texture_stone_slab,
        "COBBLESTONE": _texture_cobblestone,
        "WOOD_PLANK": _texture_wood_plank,
        "DIRT": _texture_dirt,
    }

    func = dispatch.get(floor_texture)
    if func:
        func(bm, half_x, half_y, texture_strength, floor_only_z)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def _texture_stone_slab(bm, half_x, half_y, strength, floor_only_z):
    """Rectangular stone slab pattern — 17x12mm slabs with smooth grooves."""
    slab_w = mm(17.0)
    slab_h = mm(12.0)
    groove_half = slab_w * 0.15  # 15% of slab width — wide enough for any resolution
    disp = _MAX_DISP * strength

    for v in bm.verts:
        if floor_only_z is not None and v.co.z > floor_only_z:
            continue

        # Position in slab space (offset to avoid groove at origin center)
        sx = (v.co.x + half_x) % slab_w
        sy = (v.co.y + half_y) % slab_h

        # Distance to nearest edge
        dx = min(sx, slab_w - sx)
        dy = min(sy, slab_h - sy)
        edge_dist = min(dx, dy)

        # Smooth groove falloff — works at any mesh resolution
        groove_factor = max(0.0, 1.0 - edge_dist / groove_half)
        groove_factor *= groove_factor  # Squared for smooth V-profile

        # Per-slab height variation using integer slab coords
        slab_ix = int((v.co.x + half_x) / slab_w)
        slab_iy = int((v.co.y + half_y) / slab_h)
        n = mathnoise.noise(Vector((slab_ix * 1.73, slab_iy * 2.19, 0.0)))

        v.co.z += (-groove_factor + n * 0.3 * (1.0 - groove_factor)) * disp


def _texture_cobblestone(bm, half_x, half_y, strength, floor_only_z):
    """Voronoi cell-based cobblestone pattern — rounded domes with gaps."""
    disp = _MAX_DISP * strength
    scale = 8.0

    for v in bm.verts:
        if floor_only_z is not None and v.co.z > floor_only_z:
            continue

        sample = Vector((
            (v.co.x + half_x) / (2.0 * half_x) * scale,
            (v.co.y + half_y) / (2.0 * half_y) * scale,
            0.0,
        ))

        cell = mathnoise.voronoi(sample, distance_metric='DISTANCE')
        dist_f1 = cell[0][0] if cell else 0.0

        # Dome profile: rounded stone tops, dips at boundaries
        dome = max(0.0, (1.0 - dist_f1 * 3.0))
        dome = dome * dome  # Square for smoother shape

        # Per-cell height variation
        cell_n = mathnoise.noise(Vector((sample.x * 0.5, sample.y * 0.5, 3.7)))
        variation = 0.5 + (cell_n + 1.0) * 0.25  # 0.5..1.0

        v.co.z += dome * disp * variation


def _texture_wood_plank(bm, half_x, half_y, strength, floor_only_z):
    """Parallel wood plank pattern along X axis — 6mm wide with grain."""
    plank_w = mm(6.0)
    groove_half = plank_w * 0.15  # 15% of plank width — resolution-independent
    disp = _MAX_DISP * strength

    for v in bm.verts:
        if floor_only_z is not None and v.co.z > floor_only_z:
            continue

        # Plank index from Y position
        plank_pos = (v.co.y + half_y) % plank_w
        plank_idx = int((v.co.y + half_y) / plank_w)

        # Smooth groove falloff between planks
        edge_dist = min(plank_pos, plank_w - plank_pos)
        groove_factor = max(0.0, 1.0 - edge_dist / groove_half)
        groove_factor *= groove_factor  # Squared for smooth V-profile

        # Wood grain along X at high frequency, different per plank
        grain_sample = Vector((
            (v.co.x + half_x) / (2.0 * half_x) * 20.0,
            plank_idx * 3.17,
            0.0,
        ))
        grain = mathnoise.noise(grain_sample)

        # Per-plank slight height offset
        plank_offset = mathnoise.noise(
            Vector((plank_idx * 2.31, 0.0, 5.0))
        )

        v.co.z += (-groove_factor + (grain * 0.5 + plank_offset * 0.3)
                    * (1.0 - groove_factor)) * disp


def _texture_dirt(bm, half_x, half_y, strength, floor_only_z):
    """Multi-octave fBm dirt texture — high frequency organic roughness."""
    disp = _MAX_DISP * strength

    for v in bm.verts:
        if floor_only_z is not None and v.co.z > floor_only_z:
            continue

        # Normalize position
        nx = (v.co.x + half_x) / (2.0 * half_x) if half_x > 0 else 0.5
        ny = (v.co.y + half_y) / (2.0 * half_y) if half_y > 0 else 0.5

        # High-frequency fBm
        sample_fine = Vector((nx * 15.0, ny * 15.0, 0.0))
        fbm = mathnoise.fractal(sample_fine, 0.5, 2.0, 6,
                                noise_basis='PERLIN_NEW')

        # Secondary broader turbulence
        sample_broad = Vector((nx * 4.5, ny * 4.5, 1.7))
        turb = mathnoise.turbulence(sample_broad, 3, False)

        combined = fbm * 0.7 + (turb - 0.5) * 0.6
        v.co.z += combined * disp


# ---------------------------------------------------------------------------
# Curve evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_curve_to_polyline(curve_obj):
    """Convert a curve object to an ordered list of world-space Vector points.

    Handles Bezier, NURBS, and Poly curves uniformly by converting to a
    temporary mesh via the dependency graph.

    Args:
        curve_obj: A Blender object of type CURVE

    Returns:
        List of Vector points in world space, ordered along the curve
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = curve_obj.evaluated_get(depsgraph)
    temp_mesh = bpy.data.meshes.new_from_object(eval_obj)

    if not temp_mesh.vertices:
        bpy.data.meshes.remove(temp_mesh)
        return []

    # Build adjacency from edges to walk the vertex chain in order
    adj = {}
    for edge in temp_mesh.edges:
        a, b = edge.vertices[0], edge.vertices[1]
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # Find a degree-1 endpoint to start from (open curve), or pick arbitrary
    start = 0
    for vi, neighbors in adj.items():
        if len(neighbors) == 1:
            start = vi
            break

    # Walk the chain
    ordered = [start]
    visited = {start}
    current = start
    while True:
        neighbors = adj.get(current, [])
        found = False
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                ordered.append(nb)
                current = nb
                found = True
                break
        if not found:
            break

    # Transform to world space
    mat = curve_obj.matrix_world
    points = [mat @ temp_mesh.vertices[vi].co.copy() for vi in ordered]

    bpy.data.meshes.remove(temp_mesh)
    return points


def _point_to_segment_dist(p, a, b):
    """Compute XY distance from point p to line segment a-b (Z ignored for distance).

    Args:
        p: Query point (Vector)
        a: Segment start (Vector)
        b: Segment end (Vector)

    Returns:
        (distance, closest_point, t_parameter) where t is 0..1 along the segment
    """
    ab = Vector((b.x - a.x, b.y - a.y))
    ap = Vector((p.x - a.x, p.y - a.y))
    ab_sq = ab.x * ab.x + ab.y * ab.y

    if ab_sq < 1e-12:
        dist = math.sqrt(ap.x * ap.x + ap.y * ap.y)
        return (dist, a.copy(), 0.0)

    t = max(0.0, min(1.0, (ap.x * ab.x + ap.y * ab.y) / ab_sq))
    closest = Vector((a.x + t * ab.x, a.y + t * ab.y, a.z + t * (b.z - a.z)))
    dx = p.x - closest.x
    dy = p.y - closest.y
    dist = math.sqrt(dx * dx + dy * dy)
    return (dist, closest, t)


def _closest_point_on_polyline(p, polyline):
    """Find the closest point on a polyline to point p (XY distance).

    Args:
        p: Query point (Vector)
        polyline: List of Vector points defining the polyline

    Returns:
        (min_distance, closest_point, tangent_2d) where tangent is a unit
        vector along the curve at the closest point
    """
    best_dist = float('inf')
    best_point = polyline[0].copy() if polyline else Vector((0, 0, 0))
    best_seg_idx = 0
    best_t = 0.0

    for i in range(len(polyline) - 1):
        dist, cp, t = _point_to_segment_dist(p, polyline[i], polyline[i + 1])
        if dist < best_dist:
            best_dist = dist
            best_point = cp
            best_seg_idx = i
            best_t = t

    # Compute tangent along the segment
    a = polyline[best_seg_idx]
    b = polyline[min(best_seg_idx + 1, len(polyline) - 1)]
    tang = Vector((b.x - a.x, b.y - a.y))
    length = math.sqrt(tang.x * tang.x + tang.y * tang.y)
    if length > 1e-10:
        tang /= length
    else:
        tang = Vector((1.0, 0.0))

    return (best_dist, best_point, tang)


# ---------------------------------------------------------------------------
# River channel
# ---------------------------------------------------------------------------

def apply_river_channel(obj, settings_outdoor, settings_tile):
    """Carve a river channel along a user-defined curve centerline.

    Uses cosine U-shape cross-section. Optional meander noise adds organic
    wobble perpendicular to the curve. Purely subtractive — only lowers Z.

    Args:
        obj: Blender mesh object
        settings_outdoor: TILEFORGE_PG_OutdoorSettings
        settings_tile: TILEFORGE_PG_TileSettings
    """
    if not settings_outdoor.add_river:
        return
    if settings_outdoor.river_curve is None:
        return

    polyline = _evaluate_curve_to_polyline(settings_outdoor.river_curve)
    if len(polyline) < 2:
        return

    # Transform curve world-space points into terrain object local space
    mat_inv = obj.matrix_world.inverted()
    polyline = [mat_inv @ p for p in polyline]

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    print_scale = settings_tile.print_scale
    river_half_w = m(settings_outdoor.river_width, print_scale) / 2.0
    depth = m(settings_outdoor.river_depth, print_scale)
    meander = settings_outdoor.river_meander_strength

    for v in bm.verts:
        dist, closest, tangent = _closest_point_on_polyline(v.co, polyline)

        # Meander: noise wobble shifts the effective distance
        if meander > 0.0:
            noise_sample = Vector((closest.x * 3.0, closest.y * 3.0, 0.0))
            wobble = mathnoise.noise(noise_sample)
            dist = max(0.0, dist + wobble * river_half_w * meander)

        if dist < river_half_w:
            # Cosine U-shape cross-section
            t = dist / river_half_w  # 0 at center, 1 at edge
            carve = depth * 0.5 * (1.0 + math.cos(t * math.pi))
            # Only lower Z (subtractive)
            v.co.z -= carve

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


# ---------------------------------------------------------------------------
# Path / road
# ---------------------------------------------------------------------------

def apply_path(obj, settings_outdoor, settings_tile):
    """Flatten a path along a user-defined curve centerline.

    Two-pass algorithm: first collects average height of core path vertices,
    then flattens core to 80% average + 20% original, with cosine blend edges.

    Args:
        obj: Blender mesh object
        settings_outdoor: TILEFORGE_PG_OutdoorSettings
        settings_tile: TILEFORGE_PG_TileSettings
    """
    if not settings_outdoor.add_path:
        return
    if settings_outdoor.path_curve is None:
        return

    polyline = _evaluate_curve_to_polyline(settings_outdoor.path_curve)
    if len(polyline) < 2:
        return

    # Transform curve world-space points into terrain object local space
    mat_inv = obj.matrix_world.inverted()
    polyline = [mat_inv @ p for p in polyline]

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    print_scale = settings_tile.print_scale
    path_half_w = m(settings_outdoor.path_width, print_scale) / 2.0
    blend_zone = path_half_w * 0.5  # 50% of path width for gradual transition
    total_half_w = path_half_w + blend_zone

    # Pass 1: collect core vertex heights and cache distances
    core_heights = []
    vert_dists = {}  # vert index -> distance to polyline
    for v in bm.verts:
        dist, _, _ = _closest_point_on_polyline(v.co, polyline)
        if dist <= total_half_w:
            vert_dists[v.index] = dist
            if dist <= path_half_w:
                core_heights.append(v.co.z)

    if not core_heights:
        bm.free()
        return

    avg_height = sum(core_heights) / len(core_heights)

    # Pass 2: flatten using cached distances
    for v in bm.verts:
        dist = vert_dists.get(v.index)
        if dist is None:
            continue

        if dist <= path_half_w:
            # Core: 80% average + 20% original
            v.co.z = avg_height * 0.8 + v.co.z * 0.2
        elif dist <= total_half_w:
            # Blend zone: cosine transition
            blend_t = (dist - path_half_w) / blend_zone  # 0..1
            blend_factor = 0.5 * (1.0 + math.cos(blend_t * math.pi))  # 1..0
            target = avg_height * 0.8 + v.co.z * 0.2
            v.co.z = v.co.z + (target - v.co.z) * blend_factor

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


# ---------------------------------------------------------------------------
# Heightmap-based terrain
# ---------------------------------------------------------------------------

def apply_heightmap(obj, image_path, settings_outdoor, settings_tile):
    """
    Displace vertices based on a grayscale heightmap image.
    Samples pixel brightness at each vertex's XY position.

    Args:
        obj: Blender mesh object
        image_path: Path to grayscale image file
        settings_outdoor: TILEFORGE_PG_OutdoorSettings
        settings_tile: TILEFORGE_PG_TileSettings
    """
    # Load image
    img = bpy.data.images.load(image_path, check_existing=True)
    pixels = list(img.pixels)  # Flat RGBA array
    w, h = img.size

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    print_scale = settings_tile.print_scale
    half_x = m(settings_tile.map_width, print_scale) / 2.0
    half_y = m(settings_tile.map_depth, print_scale) / 2.0
    height_range = m(settings_outdoor.terrain_height_max - settings_outdoor.terrain_height_min, print_scale)
    height_offset = m(settings_outdoor.terrain_height_min, print_scale)

    # Crop margins (percentages to 0-1 fractions)
    cl = settings_outdoor.crop_left / 100.0
    cr = settings_outdoor.crop_right / 100.0
    ct = settings_outdoor.crop_top / 100.0
    cb = settings_outdoor.crop_bottom / 100.0

    for v in bm.verts:
        # Map vertex XY to image UV (0..1)
        u_raw = (v.co.x + half_x) / (2.0 * half_x) if half_x > 0 else 0.5
        v_raw = (v.co.y + half_y) / (2.0 * half_y) if half_y > 0 else 0.5
        u_raw = max(0.0, min(1.0, u_raw))
        v_raw = max(0.0, min(1.0, v_raw))

        # Apply crop offsets
        u = cl + u_raw * (1.0 - cl - cr)
        v_coord = cb + v_raw * (1.0 - cb - ct)

        # Pixel coordinates
        px = int(u * (w - 1))
        py = int(v_coord * (h - 1))
        idx = (py * w + px) * 4  # RGBA

        # Use red channel (grayscale: R=G=B)
        brightness = pixels[idx] if idx < len(pixels) else 0.0
        brightness = max(0.0, min(1.0, brightness))

        v.co.z = height_offset + brightness * height_range

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


# ---------------------------------------------------------------------------
# Color map terrain
# ---------------------------------------------------------------------------

def _smooth_z(bm, iterations):
    """Laplacian smooth on bmesh vertex Z values only.

    Blends each vertex's Z with the average of its neighbors (50/50 mix).

    Args:
        bm: BMesh to smooth
        iterations: Number of smoothing passes
    """
    for _ in range(iterations):
        new_z = {}
        for v in bm.verts:
            neighbors = [e.other_vert(v) for e in v.link_edges]
            if neighbors:
                avg = sum(n.co.z for n in neighbors) / len(neighbors)
                new_z[v.index] = v.co.z * 0.5 + avg * 0.5
            else:
                new_z[v.index] = v.co.z
        for v in bm.verts:
            v.co.z = new_z[v.index]


def _hsv_distance_sq(h1, s1, v1, h2, s2, v2):
    """Compute weighted squared HSV distance between two HSV colors.

    Hue is weighted 2x and wraps at 1.0. Returns squared distance
    for efficient comparison against tolerance thresholds.

    Args:
        h1, s1, v1: First color in HSV (0-1 range)
        h2, s2, v2: Second color in HSV (0-1 range)

    Returns:
        Squared HSV distance (float)
    """
    # Hue wraps at 1.0
    dh = min(abs(h1 - h2), 1.0 - abs(h1 - h2))
    ds = abs(s1 - s2)
    dv = abs(v1 - v2)
    return (dh * 2.0) ** 2 + ds ** 2 + dv ** 2


def apply_color_map(obj, image_path, settings_outdoor, settings_tile):
    """Displace vertices based on color zones from an illustrated map image.

    Each pixel's RGB color is matched against user-defined color zones using
    HSV distance. The closest matching zone's height is assigned. Pixels
    matching no zone use the fallback height. Z-only Laplacian smoothing
    is applied for natural transitions.

    Args:
        obj: Blender mesh object
        image_path: Path to color map image file
        settings_outdoor: TILEFORGE_PG_OutdoorSettings
        settings_tile: TILEFORGE_PG_TileSettings
    """
    # Load image
    img = bpy.data.images.load(image_path, check_existing=True)
    pixels = list(img.pixels)  # Flat RGBA array
    w, h = img.size

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    print_scale = settings_tile.print_scale
    half_x = m(settings_tile.map_width, print_scale) / 2.0
    half_y = m(settings_tile.map_depth, print_scale) / 2.0
    height_range = m(settings_outdoor.terrain_height_max - settings_outdoor.terrain_height_min, print_scale)
    height_offset = m(settings_outdoor.terrain_height_min, print_scale)

    # Crop margins (percentages to 0-1 fractions)
    cl = settings_outdoor.crop_left / 100.0
    cr = settings_outdoor.crop_right / 100.0
    ct = settings_outdoor.crop_top / 100.0
    cb = settings_outdoor.crop_bottom / 100.0

    # Pre-convert zone colors to HSV tuples for fast access
    zones = []
    for zone in settings_outdoor.color_zones:
        zh, zs, zv = colorsys.rgb_to_hsv(zone.color[0], zone.color[1], zone.color[2])
        zones.append((
            zh, zs, zv,
            zone.tolerance ** 2,  # Store squared tolerance for comparison
            zone.height,
        ))

    fallback = settings_outdoor.fallback_height

    for v in bm.verts:
        # Map vertex XY to image UV (0..1)
        u_raw = (v.co.x + half_x) / (2.0 * half_x) if half_x > 0 else 0.5
        v_raw = (v.co.y + half_y) / (2.0 * half_y) if half_y > 0 else 0.5
        u_raw = max(0.0, min(1.0, u_raw))
        v_raw = max(0.0, min(1.0, v_raw))

        # Apply crop offsets
        u = cl + u_raw * (1.0 - cl - cr)
        v_coord = cb + v_raw * (1.0 - cb - ct)

        # Pixel coordinates
        px = int(u * (w - 1))
        py = int(v_coord * (h - 1))
        idx = (py * w + px) * 4  # RGBA

        # Sample pixel RGB and convert to HSV once per vertex
        if idx + 2 < len(pixels):
            pr, pg, pb = pixels[idx], pixels[idx + 1], pixels[idx + 2]
        else:
            pr, pg, pb = 0.0, 0.0, 0.0
        ph, ps, pv = colorsys.rgb_to_hsv(pr, pg, pb)

        # Find closest matching zone
        best_dist = float('inf')
        best_height = fallback
        best_tol_sq = 0.0
        for zh, zs, zv, tol_sq, z_height in zones:
            dist = _hsv_distance_sq(ph, ps, pv, zh, zs, zv)
            if dist < best_dist:
                best_dist = dist
                best_height = z_height
                best_tol_sq = tol_sq

        # If best match exceeds its tolerance, use fallback
        if best_dist > best_tol_sq:
            best_height = fallback

        v.co.z = height_offset + best_height * height_range

    # Apply Z-only Laplacian smoothing
    if settings_outdoor.map_smoothing > 0:
        _smooth_z(bm, settings_outdoor.map_smoothing)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


# ---------------------------------------------------------------------------
# Shade smooth
# ---------------------------------------------------------------------------

def apply_shade_smooth(obj):
    """Apply smooth shading to terrain faces for higher quality surface.

    Only smooths faces whose normal points mostly upward (terrain surface).
    Side walls and bottom stay flat-shaded for clean print edges.

    Args:
        obj: Blender mesh object
    """
    mesh = obj.data
    mesh.shade_smooth()

    # Use auto-smooth so that sharp edges (side walls, base) stay crisp
    # while the terrain surface gets interpolated normals
    if not mesh.has_custom_normals:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        # Mark edges with a sharp angle change (side walls, base perimeter)
        for edge in bm.edges:
            if len(edge.link_faces) == 2:
                angle = edge.calc_face_angle(0.0)
                # Edges steeper than 60 degrees stay sharp
                if angle > math.radians(60):
                    edge.smooth = False
                else:
                    edge.smooth = True
            else:
                edge.smooth = False
        bm.to_mesh(mesh)
        bm.free()

    mesh.update()


# ---------------------------------------------------------------------------
# Terrain material
# ---------------------------------------------------------------------------

def assign_terrain_material(obj):
    """
    Create or reuse a Principled BSDF material with earth-toned color
    and assign it to the object.
    """
    mat_name = "TileForge_Terrain"
    mat = bpy.data.materials.get(mat_name)

    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            # Warm brown earth tone
            bsdf.inputs["Base Color"].default_value = (0.36, 0.25, 0.15, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.8

    obj.data.materials.clear()
    obj.data.materials.append(mat)


# ---------------------------------------------------------------------------
# Solid base extrusion
# ---------------------------------------------------------------------------

def add_solid_base(obj, base_height_mm):
    """
    Extrude the terrain mesh downward to create a solid base.
    Duplicates the entire top surface at bottom_z with reversed winding,
    then connects the perimeter with side walls for a fully watertight volume.

    Args:
        obj: Blender mesh object with terrain on Z
        base_height_mm: Base thickness in mm (physical print dimension)
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    # Snapshot original geometry before modifying
    original_verts = list(bm.verts)
    original_faces = list(bm.faces)
    boundary_edges = [e for e in bm.edges if e.is_boundary]

    # Find minimum Z to know where bottom should be
    min_z = min(v.co.z for v in original_verts)
    bottom_z = min_z - mm(base_height_mm)

    # Duplicate ALL verts at bottom Z (not just boundary)
    vert_map = {}
    for v in original_verts:
        new_v = bm.verts.new((v.co.x, v.co.y, bottom_z))
        vert_map[v] = new_v

    bm.verts.ensure_lookup_table()

    # Create bottom faces — mirror every top face with reversed winding
    for face in original_faces:
        bottom_face_verts = [vert_map[v] for v in reversed(face.verts)]
        try:
            bm.faces.new(bottom_face_verts)
        except ValueError:
            pass

    # Create side walls from boundary edges with correct outward winding.
    # A boundary edge has exactly one adjacent face. We check the vertex
    # order in that face to determine which winding produces an outward normal.
    for e in boundary_edges:
        v1, v2 = e.verts
        b1 = vert_map[v1]
        b2 = vert_map[v2]

        face = e.link_faces[0]
        face_verts = list(face.verts)
        i1 = face_verts.index(v1)
        # If v1→v2 follows the face's CCW winding, the outward side wall
        # must go in the opposite direction: v2→v1 at top, b1→b2 at bottom.
        if face_verts[(i1 + 1) % len(face_verts)] == v2:
            quad = [v2, v1, b1, b2]
        else:
            quad = [v1, v2, b2, b1]
        try:
            bm.faces.new(quad)
        except ValueError:
            pass

    # Final safety pass — recalculate normals to ensure consistency
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


# ---------------------------------------------------------------------------
# Grid line engraving
# ---------------------------------------------------------------------------

def engrave_grid_lines(obj, settings_tile):
    """
    Engrave grid lines into the top surface of a tile by bisecting the mesh
    at groove boundaries and displacing surface vertices downward.

    This follows the terrain contour at any height, unlike a flat boolean
    cutter which only intersects near the peak.

    Args:
        obj: The tile mesh object (already sliced, centered at origin)
        settings_tile: TILEFORGE_PG_TileSettings
    """
    if not settings_tile.engrave_grid:
        return

    print_scale = settings_tile.print_scale
    tile_x = m(settings_tile.tile_size_x, print_scale)
    tile_y = m(settings_tile.tile_size_y, print_scale)
    grid_squares = settings_tile.grid_squares
    grid_spacing_x = tile_x / grid_squares
    # Keep Y cells as square as possible
    grid_squares_y = max(1, round(settings_tile.tile_size_y / (settings_tile.tile_size_x / grid_squares)))
    grid_spacing_y = tile_y / grid_squares_y
    half_w = mm(settings_tile.grid_line_width) / 2.0
    line_depth = mm(settings_tile.grid_line_depth)

    mesh_data = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh_data)

    # --- Bisect at every groove edge to create precise boundaries ----------
    # Vertical grid lines (cuts perpendicular to X)
    for i in range(grid_squares + 1):
        x = -tile_x / 2.0 + i * grid_spacing_x
        for offset in (-half_w, half_w):
            geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
            bmesh.ops.bisect_plane(
                bm, geom=geom,
                plane_co=(x + offset, 0.0, 0.0),
                plane_no=(1.0, 0.0, 0.0),
            )

    # Horizontal grid lines (cuts perpendicular to Y)
    for i in range(grid_squares_y + 1):
        y = -tile_y / 2.0 + i * grid_spacing_y
        for offset in (-half_w, half_w):
            geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
            bmesh.ops.bisect_plane(
                bm, geom=geom,
                plane_co=(0.0, y + offset, 0.0),
                plane_no=(0.0, 1.0, 0.0),
            )

    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.normal_update()

    # --- Displace surface vertices inside groove strips --------------------
    # Base vertices sit at the mesh minimum Z; skip them.
    min_z = min(v.co.z for v in bm.verts)
    base_threshold = min_z + mm(0.1)

    # Pre-compute grid line center positions
    x_lines = [-tile_x / 2.0 + i * grid_spacing_x for i in range(grid_squares + 1)]
    y_lines = [-tile_y / 2.0 + i * grid_spacing_y for i in range(grid_squares_y + 1)]

    eps = half_w + 1e-7  # small tolerance for bisect precision

    for v in bm.verts:
        if v.co.z <= base_threshold:
            continue

        # Skip side-wall vertices: any linked face with a near-vertical normal
        # (normal.z close to 0) means this vertex is on a wall, not the surface.
        is_side_wall = False
        for f in v.link_faces:
            if abs(f.normal.z) < 0.1:
                is_side_wall = True
                break
        if is_side_wall:
            continue

        in_groove = False
        for lx in x_lines:
            if abs(v.co.x - lx) <= eps:
                in_groove = True
                break
        if not in_groove:
            for ly in y_lines:
                if abs(v.co.y - ly) <= eps:
                    in_groove = True
                    break

        if in_groove:
            v.co.z -= line_depth

    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update()


# ---------------------------------------------------------------------------
# Connectors (Peg & Slot)
# ---------------------------------------------------------------------------

def add_connectors(obj, settings_tile, tile_col, tile_row):
    """
    Add interlocking connectors to a tile based on its position in the grid.
    Pegs on South (+Y) and East (+X) edges.
    Slots on North (-Y) and West (-X) edges.

    Args:
        obj: Tile mesh object
        settings_tile: TILEFORGE_PG_TileSettings
        tile_col: Column index (0-based)
        tile_row: Row index (0-based)
    """
    if settings_tile.connector_type == 'NONE':
        return

    print_scale = settings_tile.print_scale
    tile_x = m(settings_tile.tile_size_x, print_scale)
    tile_y = m(settings_tile.tile_size_y, print_scale)
    radius = mm(settings_tile.connector_diameter) / 2.0
    tolerance = mm(settings_tile.connector_tolerance)
    height = mm(settings_tile.connector_height)

    num_connectors = 2  # Two connectors per edge

    max_cols = settings_tile.map_tiles_x
    max_rows = settings_tile.map_tiles_y

    # East edge: pegs if not rightmost column
    if tile_col < max_cols - 1:
        for i in range(num_connectors):
            y_pos = -tile_y / 2.0 + tile_y * (i + 1) / (num_connectors + 1)
            _add_peg(obj, (tile_x / 2.0, y_pos), radius, height, axis='X')

    # West edge: slots if not leftmost column
    if tile_col > 0:
        for i in range(num_connectors):
            y_pos = -tile_y / 2.0 + tile_y * (i + 1) / (num_connectors + 1)
            _add_slot(obj, (-tile_x / 2.0, y_pos), radius + tolerance, height, axis='X')

    # North edge: pegs if not top row
    if tile_row < max_rows - 1:
        for i in range(num_connectors):
            x_pos = -tile_x / 2.0 + tile_x * (i + 1) / (num_connectors + 1)
            _add_peg(obj, (x_pos, tile_y / 2.0), radius, height, axis='Y')

    # South edge: slots if not bottom row
    if tile_row > 0:
        for i in range(num_connectors):
            x_pos = -tile_x / 2.0 + tile_x * (i + 1) / (num_connectors + 1)
            _add_slot(obj, (x_pos, -tile_y / 2.0), radius + tolerance, height, axis='Y')


def _add_peg(obj, position_xy, radius, height, axis='X'):
    """Add a cylindrical peg protruding from an edge."""
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=False,
        segments=16,
        radius1=radius,
        radius2=radius,
        depth=height,
    )

    # Rotate to point along correct axis
    if axis == 'X':
        for v in bm.verts:
            x, y, z = v.co
            v.co = Vector((z + position_xy[0], position_xy[1], 0))
    else:
        for v in bm.verts:
            x, y, z = v.co
            v.co = Vector((position_xy[0], z + position_xy[1], 0))

    # Shift Z to base level
    min_z = min(v.co.z for v in bm.verts)
    for v in bm.verts:
        v.co.z -= min_z

    peg_mesh = bpy.data.meshes.new("peg_temp")
    bm.to_mesh(peg_mesh)
    bm.free()

    peg_obj = bpy.data.objects.new("peg_temp", peg_mesh)
    bpy.context.collection.objects.link(peg_obj)

    # Join with main object (deselect all first to avoid absorbing other tiles)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    peg_obj.select_set(True)
    bpy.ops.object.join()


def _add_slot(obj, position_xy, radius, depth, axis='X'):
    """Cut a cylindrical slot into an edge using boolean difference."""
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=False,
        segments=16,
        radius1=radius,
        radius2=radius,
        depth=depth * 2,  # Extra depth to ensure clean cut
    )

    if axis == 'X':
        for v in bm.verts:
            x, y, z = v.co
            v.co = Vector((z + position_xy[0], position_xy[1], 0))
    else:
        for v in bm.verts:
            x, y, z = v.co
            v.co = Vector((position_xy[0], z + position_xy[1], 0))

    min_z = min(v.co.z for v in bm.verts)
    for v in bm.verts:
        v.co.z -= min_z

    slot_mesh = bpy.data.meshes.new("slot_temp")
    bm.to_mesh(slot_mesh)
    bm.free()

    slot_obj = bpy.data.objects.new("slot_temp", slot_mesh)
    bpy.context.collection.objects.link(slot_obj)

    mod = obj.modifiers.new(name="SlotCut", type='BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.object = slot_obj
    mod.solver = 'EXACT'

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    bpy.data.objects.remove(slot_obj, do_unlink=True)


# ---------------------------------------------------------------------------
# Dungeon generation
# ---------------------------------------------------------------------------

def generate_dungeon_tile(name, settings_dungeon, settings_tile, tile_col, tile_row):
    """
    Generate a dungeon tile from a floor plan image.
    White pixels = floor, Black pixels = walls.

    Args:
        name: Object name
        settings_dungeon: TILEFORGE_PG_DungeonSettings
        settings_tile: TILEFORGE_PG_TileSettings
        tile_col: Column index in the map grid
        tile_row: Row index in the map grid

    Returns:
        Blender object
    """
    print_scale = settings_tile.print_scale
    tile_x = m(settings_tile.tile_size_x, print_scale)
    tile_y = m(settings_tile.tile_size_y, print_scale)
    wall_h = m(settings_dungeon.wall_height, print_scale)
    base_h = mm(settings_tile.base_height)  # Physical print dimension

    # Load floor plan image
    img = bpy.data.images.load(settings_dungeon.floorplan_image, check_existing=True)
    pixels = list(img.pixels)
    img_w, img_h = img.size

    # Calculate which portion of the image this tile covers
    total_w = settings_tile.map_tiles_x
    total_h = settings_tile.map_tiles_y

    u_start = tile_col / total_w
    u_end = (tile_col + 1) / total_w
    v_start = tile_row / total_h
    v_end = (tile_row + 1) / total_h

    # Create base grid at higher resolution for wall detection
    resolution = 64
    obj, mesh = create_base_grid(
        name, settings_tile.tile_size_x, settings_tile.tile_size_y,
        resolution, print_scale,
    )

    bm = bmesh.new()
    bm.from_mesh(mesh)

    # Sample image to determine floor vs wall for each vertex
    half_x = tile_x / 2.0
    half_y = tile_y / 2.0

    wall_verts = set()
    for v in bm.verts:
        # Map vertex position to image UV
        local_u = (v.co.x + half_x) / tile_x  # 0..1 within tile
        local_v = (v.co.y + half_y) / tile_y

        # Map to global image UV
        img_u = u_start + local_u * (u_end - u_start)
        img_v = v_start + local_v * (v_end - v_start)

        img_u = max(0.0, min(1.0, img_u))
        img_v = max(0.0, min(1.0, img_v))

        px = int(img_u * (img_w - 1))
        py = int(img_v * (img_h - 1))
        idx = (py * img_w + px) * 4

        brightness = pixels[idx] if idx < len(pixels) else 1.0

        # Dark = wall, light = floor
        if brightness < 0.5:
            wall_verts.add(v.index)
            v.co.z = wall_h
        else:
            v.co.z = 0.0  # Floor level (base will be added below)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    return obj


# ---------------------------------------------------------------------------
# Tile slicing
# ---------------------------------------------------------------------------

def slice_terrain_to_tile(source_obj, settings_tile, tile_col, tile_row):
    """
    Extract a single tile from a larger terrain mesh using a boolean intersection.

    Args:
        source_obj: The full terrain mesh
        settings_tile: TILEFORGE_PG_TileSettings
        tile_col: Column index
        tile_row: Row index

    Returns:
        New object containing just this tile's geometry
    """
    print_scale = settings_tile.print_scale
    tile_x = m(settings_tile.tile_size_x, print_scale)
    tile_y = m(settings_tile.tile_size_y, print_scale)

    # Calculate tile center position from map dimensions
    total_x = m(settings_tile.map_width, print_scale)
    total_y = m(settings_tile.map_depth, print_scale)

    center_x = -total_x / 2.0 + tile_x * (tile_col + 0.5)
    center_y = -total_y / 2.0 + tile_y * (tile_row + 0.5)

    # Create a cutter box for this tile
    cutter_mesh = bpy.data.meshes.new("tile_cutter_mesh")
    cutter_obj = bpy.data.objects.new("tile_cutter", cutter_mesh)

    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)

    # Scale and position — derive cutter height from actual mesh extents
    src_verts = source_obj.data.vertices
    min_z = min(v.co.z for v in src_verts)
    max_z = max(v.co.z for v in src_verts)
    margin = max(max_z - min_z, mm(100.0)) * 0.1  # 10% margin
    cutter_min_z = min_z - margin
    cutter_max_z = max_z + margin
    cutter_mid_z = (cutter_min_z + cutter_max_z) / 2.0
    cutter_height = cutter_max_z - cutter_min_z

    for v in bm.verts:
        v.co.x = center_x + v.co.x * tile_x
        v.co.y = center_y + v.co.y * tile_y
        v.co.z = cutter_mid_z + v.co.z * cutter_height

    bm.to_mesh(cutter_mesh)
    bm.free()

    bpy.context.collection.objects.link(cutter_obj)

    # Duplicate source object
    tile_obj = source_obj.copy()
    tile_obj.data = source_obj.data.copy()
    tile_obj.name = f"tile_r{tile_row}_c{tile_col}"
    bpy.context.collection.objects.link(tile_obj)

    # Boolean intersection
    mod = tile_obj.modifiers.new(name="TileSlice", type='BOOLEAN')
    mod.operation = 'INTERSECT'
    mod.object = cutter_obj
    mod.solver = 'EXACT'

    bpy.context.view_layer.objects.active = tile_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    # Cleanup cutter
    bpy.data.objects.remove(cutter_obj, do_unlink=True)

    # Re-center mesh data at origin (not just object location)
    # so that connectors and grid engraving align correctly
    tile_mesh = tile_obj.data
    bm = bmesh.new()
    bm.from_mesh(tile_mesh)
    for v in bm.verts:
        v.co.x -= center_x
        v.co.y -= center_y
    bm.to_mesh(tile_mesh)
    bm.free()
    tile_mesh.update()

    return tile_obj


# ---------------------------------------------------------------------------
# Mesh validation
# ---------------------------------------------------------------------------

def check_manifold(obj):
    """
    Check if mesh is manifold (watertight).

    Returns:
        (is_manifold: bool, problem_edges: int, problem_verts: int)
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    non_manifold_edges = [e for e in bm.edges if not e.is_manifold]
    non_manifold_verts = [v for v in bm.verts if not v.is_manifold]

    result = (
        len(non_manifold_edges) == 0 and len(non_manifold_verts) == 0,
        len(non_manifold_edges),
        len(non_manifold_verts),
    )

    bm.free()
    return result
