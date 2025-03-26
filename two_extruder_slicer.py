import trimesh
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import json
from PIL import Image
import os
import io

# Optional: For potential visualization or advanced 2D geometry processing
# import matplotlib.pyplot as plt
# from shapely.geometry import Polygon, MultiPolygon
# from trimesh.path.polygons import paths_to_polygons # Helper

INPUT_GLB_FILE = Path("input.glb")
OUTPUT_PLAN_FILE = Path("slicing_plan.json")
LAYER_HEIGHT = 0.2  # mm (Adjust based on your printer/desired quality)
FILAMENT_DIAMETER = 1.75 # mm (Used for context, not direct calculation here)

# --- Functions ---
def identify_and_assign_colors_texture(mesh, num_colors=2):
    """
    Identifies dominant colors by sampling the mesh's texture map based on UV coordinates,
    clusters these colors using K-Means, and assigns each face to a color index.

    Args:
        mesh (trimesh.Trimesh): The input mesh object, expected to have UVs and a texture.
        num_colors (int): The expected number of distinct colors (extruders).

    Returns:
        tuple: (
            face_color_indices (np.ndarray): Array mapping face index to cluster index (0 or 1).
            cluster_centers (np.ndarray): The representative RGB colors (0-255) for each cluster.
        )
        Returns None, None if required visual information (UVs, texture) is missing or invalid.
    """
    print("Analyzing mesh texture colors...")

    # --- Pre-requisite Checks ---
    if not isinstance(mesh, trimesh.Trimesh):
        print("Error: Input is not a valid Trimesh object.")
        return None, None
    if not mesh.visual or not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        print("Error: Mesh does not have texture coordinates (UVs) defined in mesh.visual.uv.")
        return None, None
    if not hasattr(mesh.visual, 'material') or not mesh.visual.material:
        print("Error: Mesh does not have a material defined.")
        return None, None

    material = mesh.visual.material
    texture_image = None

    # Try to get the base color texture (most common case)
    if hasattr(material, 'baseColorTexture') and isinstance(material.baseColorTexture, Image.Image):
        texture_image = material.baseColorTexture
        print("Found baseColorTexture.")
    # Fallback: Check other common texture attributes if needed (e.g., PBR metallicRoughness)
    # elif hasattr(material, 'diffuseTexture') ... etc.
    else:
        # Sometimes trimesh stores texture image directly in visual if no complex material
         if hasattr(mesh.visual, 'material_image') and isinstance(mesh.visual.material_image, Image.Image):
              texture_image = mesh.visual.material_image
              print("Found material_image on mesh.visual.")

    if texture_image is None:
        print("Error: Could not find a suitable texture image associated with the material.")
        # You might need to inspect the specific material type (SimpleMaterial, PBRMaterial)
        # and its attributes depending on how the GLB was exported.
        print(f"Material type: {type(material)}")
        # print(f"Material attributes: {dir(material)}") # Uncomment to debug material properties
        return None, None

    # --- Sample Texture Colors for Each Face ---
    print("Sampling texture colors for each face...")
    try:
        # Calculate the average UV coordinate for each face center.
        # mesh.visual.uv holds UVs per vertex: shape (n_vertices, 2)
        # mesh.faces holds vertex indices per face: shape (n_faces, 3)
        # We get UVs for each vertex of each face: shape (n_faces, 3, 2)
        face_uvs = mesh.visual.uv[mesh.faces]
        # Average the UVs for the 3 vertices of each face: shape (n_faces, 2)
        face_uv_centers = face_uvs.mean(axis=1)

        # Use trimesh's built-in function to sample colors at the given UV coordinates
        # This handles the texture lookup and interpolation.
        # It expects UVs, the mesh itself (to know which faces use which UVs), and the image.
        # Note: uv_to_color might have slightly different args depending on trimesh version.
        # Let's try passing the explicit image if available.
        # We need to ensure the mesh's visual has the image assigned correctly for uv_to_color
        # If uv_to_color doesn't work directly, manual sampling is needed.

        # Check if the image is already linked correctly within trimesh's visual structure
        if not (hasattr(mesh.visual, '_material_image_cache') and texture_image == mesh.visual._material_image_cache.get(mesh.visual.material.name)):
             # Manually associate the found image if uv_to_color needs it implicitly
             # This is a bit of a workaround, hoping uv_to_color picks it up
             mesh.visual.material.baseColorTexture = texture_image
             # mesh.visual.material_image = texture_image # Try assigning here too

        print(f"Attempting to sample {len(face_uv_centers)} face UV centers...")
        sampled_colors_rgba = mesh.visual.uv_to_color(face_uv_centers)

        if sampled_colors_rgba is None:
             print("Error: mesh.visual.uv_to_color returned None. Sampling failed.")
             # Fallback: Manual sampling (more complex)
             # You would need to convert UVs to pixel coords and use texture_image.getpixel()
             print("Manual texture sampling not implemented in this example.")
             return None, None

        print(f"Sampled colors shape: {sampled_colors_rgba.shape}") # Should be (n_faces, 4)

    except AttributeError as e:
        print(f"Error during texture sampling (likely missing visual components): {e}")
        print("Ensure the GLB has UVs and the texture image is loaded correctly by trimesh.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during texture sampling: {e}")
        return None, None

    # --- Cluster the Sampled Colors ---
    print(f"Clustering {len(sampled_colors_rgba)} sampled face colors into {num_colors} groups...")

    # Ensure colors are RGB and normalized (0-1) for KMeans
    if sampled_colors_rgba.shape[1] == 4:
        colors_rgb = sampled_colors_rgba[:, :3]
    elif sampled_colors_rgba.shape[1] == 3:
        colors_rgb = sampled_colors_rgba
    else:
        print(f"Error: Unexpected sampled color format shape: {sampled_colors_rgba.shape}")
        return None, None

    # Normalize colors if they are 0-255 (uv_to_color usually returns 0-255)
    if colors_rgb.max() > 1.0:
        print("Normalizing sampled colors from 0-255 range to 0-1 range.")
        colors_rgb_normalized = colors_rgb / 255.0
    else:
        colors_rgb_normalized = colors_rgb # Assume already normalized

    unique_colors = np.unique(colors_rgb_normalized.round(decimals=5), axis=0) # Round to avoid floating point issues
    print(f"Found {len(unique_colors)} unique sampled face colors (approx).")

    if len(unique_colors) < num_colors:
        print(f"Warning: Expected {num_colors} dominant colors, but found only {len(unique_colors)} unique sampled colors.")
        # Decide how to proceed: Maybe assign all to extruder 0? Or error out?
        # Let's try assigning based on the limited unique colors if possible.
        if len(unique_colors) == 0: return None, None # No colors!
        if len(unique_colors) == 1 and num_colors > 1:
             print("Assigning all faces to extruder 0.")
             face_color_indices = np.zeros(len(mesh.faces), dtype=int)
             # Return the single color found as both cluster centers (not ideal, but provides output)
             cluster_centers_normalized = np.vstack([unique_colors[0], unique_colors[0]])
             return face_color_indices, cluster_centers_normalized * 255.0

        # If len(unique_colors) < num_colors but > 1, KMeans might still work or fail.

    try:
        kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init=10).fit(colors_rgb_normalized)
        cluster_centers_normalized = kmeans.cluster_centers_

        # Assign each *sampled face color* to the nearest cluster center
        print("Assigning faces to color clusters based on sampled texture...")
        face_color_indices = kmeans.predict(colors_rgb_normalized)
    except Exception as e:
        print(f"Error during K-Means clustering on sampled colors: {e}")
        return None, None

    print(f"Identified {num_colors} dominant color clusters from texture.")
    print("Representative cluster colors (RGB 0-255 approx):")
    cluster_centers_rgb255 = cluster_centers_normalized * 255.0
    for i, color in enumerate(cluster_centers_rgb255):
        print(f"  Extruder {i}: [{int(color[0])}, {int(color[1])}, {int(color[2])}]")

    return face_color_indices, cluster_centers_rgb255faaaaa

def create_meshes_for_extruders(mesh, face_color_indices, num_colors):
    """
    Creates separate mesh objects for each color/extruder.
    """
    print("Separating mesh geometry by assigned color...")
    meshes_by_extruder = []
    # Ensure vertices are copied, faces reference the *original* vertex indices
    vertices = mesh.vertices.copy()

    for i in range(num_colors):
        extruder_face_mask = (face_color_indices == i)
        extruder_faces = mesh.faces[extruder_face_mask]
        if len(extruder_faces) > 0:
            # Create a new mesh for this extruder's faces
            # Important: It shares vertices with the original mesh for now
            extruder_mesh = trimesh.Trimesh(vertices=vertices, faces=extruder_faces)
            # Optional: Clean mesh to remove unused vertices (makes it self-contained)
            # extruder_mesh.remove_unreferenced_vertices() # Be careful if slicing relies on original indices
            meshes_by_extruder.append(extruder_mesh)
            print(f"  Created mesh for Extruder {i} with {len(extruder_faces)} faces.")
        else:
            print(f"  Warning: No faces assigned to Extruder {i}.")
            meshes_by_extruder.append(None) # Placeholder if an extruder has no geometry

    return meshes_by_extruder

def generate_slice_plan(meshes_by_extruder, layer_height):
    """
    Slices each extruder's mesh and compiles a layer-by-layer plan.
    """
    print(f"\nSlicing models with layer height: {layer_height}mm...")
    num_extruders = len(meshes_by_extruder)
    slicing_plan = {}

    # Determine the Z-range for slicing across all meshes that exist
    min_z = float('inf')
    max_z = float('-inf')
    valid_meshes_found = False
    for mesh in meshes_by_extruder:
        if mesh is not None and len(mesh.faces) > 0:
            min_z = min(min_z, mesh.bounds[0, 2])
            max_z = max(max_z, mesh.bounds[1, 2])
            valid_meshes_found = True

    if not valid_meshes_found:
        print("Error: No valid mesh geometry found to slice.")
        return None

    # Define the Z heights for slicing
    # Start slightly above the min Z to catch the first layer properly
    z_levels = np.arange(min_z + layer_height / 2.0, max_z, layer_height)
    print(f"Total layers: {len(z_levels)}")

    if len(z_levels) == 0:
        print("Warning: No layers generated. Check model height and layer height.")
        return {}

    for i, z in enumerate(z_levels):
        print(f"\r  Slicing layer {i+1}/{len(z_levels)} at Z = {z:.3f}mm", end="")
        layer_data = {}
        plane_origin = [0, 0, z]
        plane_normal = [0, 0, 1]

        for extruder_index, mesh in enumerate(meshes_by_extruder):
            extruder_key = f"extruder_{extruder_index}"
            if mesh is not None and len(mesh.faces) > 0:
                try:
                    # Perform the slice
                    section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)

                    if section is not None and len(section.entities) > 0:
                        # Convert the 2D path (lines) into closed polygons (loops)
                        # Trimesh sections can be complex (multiple disjoint loops)
                        # 'polygons_closed' contains numpy arrays of vertices for each loop
                        polygons, _ = trimesh.path.exchange.misc.paths_to_polygons(section.entities)

                        # Store polygon data (list of vertex coordinates for each loop)
                        # Convert to simple lists for JSON serialization
                        layer_data[extruder_key] = [[vertex.tolist() for vertex in poly] for poly in polygons]
                    else:
                        layer_data[extruder_key] = [] # No geometry for this extruder at this layer
                except Exception as e:
                    print(f"\nWarning: Error slicing mesh for extruder {extruder_index} at Z={z:.3f}: {e}")
                    layer_data[extruder_key] = []
            else:
                 layer_data[extruder_key] = [] # No mesh for this extruder

        # Only add layer to plan if at least one extruder has geometry
        if any(layer_data.values()):
            slicing_plan[f"{z:.4f}"] = layer_data # Use Z height as key (string for JSON)
        # else: print(f"\n  Skipping empty layer at Z = {z:.3f}mm")


    print("\nSlicing complete.")
    return slicing_plan


'''
# --- Main Execution ---
if __name__ == "__main__":
    if not INPUT_GLB_FILE.is_file():
        print(f"Error: Input file not found at {INPUT_GLB_FILE}")
        exit(1)

    print(f"Loading GLB file: {INPUT_GLB_FILE}")
    try:
        # force='mesh' ensures we get a single mesh, potentially merging geometry
        # process=True helps clean up the mesh
        mesh = trimesh.load(INPUT_GLB_FILE, force='mesh', process=True)
        print("GLB loaded successfully.")
        print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
    except Exception as e:
        print(f"Error loading GLB file: {e}")
        exit(1)

    # Make sure it's a usable mesh
    if not isinstance(mesh, trimesh.Trimesh):
         print(f"Error: Loaded object is not a Trimesh mesh (type: {type(mesh)}). It might be a Scene.")
         # If it's a scene, you might need to iterate through mesh.geometry.values()
         # and merge them or handle them individually if they represent different parts.
         # For simplicity, this script assumes a single logical mesh.
         print("Attempting to extract first mesh from scene...")
         if isinstance(mesh, trimesh.Scene) and mesh.geometry:
             mesh = list(mesh.geometry.values())[0] # Take the first mesh found
             if not isinstance(mesh, trimesh.Trimesh):
                  print("Error: Could not extract a valid Trimesh object from the scene.")
                  exit(1)
             print("Proceeding with the first mesh found in the scene.")
         else:
             print("Error: Cannot proceed.")
             exit(1)


    # 1. Identify colors and assign faces to extruders
    face_color_indices, cluster_centers = identify_and_assign_colors(mesh, num_colors=2)

    if face_color_indices is None:
        print("Could not determine color assignments. Exiting.")
        exit(1)

    # 2. Create separate meshes based on color assignment
    meshes_by_extruder = create_meshes_for_extruders(mesh, face_color_indices, num_colors=2)

    # 3. Slice the meshes layer by layer
    slicing_plan = generate_slice_plan(meshes_by_extruder, LAYER_HEIGHT)

    # 4. Save the slicing plan
    if slicing_plan is not None:
        print(f"Saving slicing plan to: {OUTPUT_PLAN_FILE}")
        try:
            with open(OUTPUT_PLAN_FILE, 'w') as f:
                json.dump(slicing_plan, f, indent=2)
            print("Slicing plan saved successfully.")
        except Exception as e:
            print(f"Error saving slicing plan to JSON: {e}")

    print("\nProcess finished.")
    print("\n--- Summary ---")
    print(f"Input GLB: {INPUT_GLB_FILE}")
    print(f"Output Plan: {OUTPUT_PLAN_FILE}")
    print(f"Layer Height: {LAYER_HEIGHT}mm")
    print(f"Detected Colors (approx RGB 0-255):")
    if cluster_centers is not None:
        for i, color in enumerate(cluster_centers):
             print(f"  Extruder {i}: [{int(color[0])}, {int(color[1])}, {int(color[2])}]")
    if slicing_plan is not None:
         print(f"Generated plan for {len(slicing_plan)} layers.")
    print("---------------")
    print("\nNOTE: This script generates a geometric plan (polygons per layer per extruder).")
    print("It does NOT generate printable G-code. You would need another process to:")
    print("  - Generate infill for the polygons.")
    print("  - Create toolpaths (perimeter, infill lines).")
    print("  - Handle tool changes (purge towers/blocks, wipe sequences).")
    print("  - Add commands for temperature, speed, retraction, cooling, etc.")
    print("  - Generate travel moves.")
    print("This output is primarily for analysis or as input to a custom G-code generator.")
'''



def get_face_colors_from_texture(mesh):
    """
    Samples the texture color for each face based on UV coordinates.
    Assumes TextureVisuals and a baseColorTexture.
    Returns a numpy array of RGB colors, one per face.
    """
    if not isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        raise TypeError("Mesh visual is not TextureVisuals.")
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
         raise ValueError("Mesh does not have UV coordinates.")
    if not hasattr(mesh.visual, 'material') or \
       not hasattr(mesh.visual.material, 'baseColorTexture') or \
       mesh.visual.material.baseColorTexture is None:
        raise ValueError("Mesh material does not have a baseColorTexture.")

    texture = mesh.visual.material.baseColorTexture
    if not isinstance(texture, Image.Image):
         # Sometimes trimesh might store texture differently, try to load if it's bytes
         try:
             # Assuming it might be raw data stored in material PBR properties
             # This part might need adjustment based on how trimesh loads your specific GLB
             if hasattr(mesh.visual.material, 'pbrMetallicRoughness') and \
                hasattr(mesh.visual.material.pbrMetallicRoughness, 'baseColorTexture') and \
                'source' in mesh.visual.material.pbrMetallicRoughness.baseColorTexture:
                  tex_info = mesh.visual.material.pbrMetallicRoughness.baseColorTexture['source']
                  # Need to find the image data in the gltf resources
                  # This is complex and depends on the GLTF structure trimesh exposes
                  # A simpler trimesh load might directly give a PIL image.
                  # Let's assume mesh.visual.material.baseColorTexture IS a PIL Image for now.
                  # If not, this part needs significant GLTF structure diving.
                  raise NotImplementedError("Texture loading from GLTF structure not fully implemented here.")
             else:
                raise ValueError("Cannot access texture image data.")
         except Exception as e:
             raise ValueError(f"Could not interpret texture: {e}")

    # Ensure texture is RGB
    if texture.mode == 'RGBA':
        # Create a white background image
        bg = Image.new('RGB', texture.size, (255, 255, 255))
        # Paste the RGBA image onto the white background
        bg.paste(texture, (0, 0), texture)
        texture = bg
    elif texture.mode != 'RGB':
        texture = texture.convert('RGB')

    tex_pixels = np.array(texture)
    tex_height, tex_width, _ = tex_pixels.shape

    uv = mesh.visual.uv
    face_colors = []

    for face in mesh.faces:
        # Get UV coordinates for the vertices of the face
        uv_face = uv[face] # Shape: (3, 2)

        # Calculate the centroid UV coordinate for the face
        # This is a simple approximation for the face's color
        uv_centroid = uv_face.mean(axis=0)

        # Convert UV coordinates (0-1 range) to pixel coordinates
        # UV origin (0,0) is often bottom-left, Image origin (0,0) is top-left
        px = int(uv_centroid[0] * (tex_width - 1))
        py = int((1.0 - uv_centroid[1]) * (tex_height - 1)) # Invert Y for image coords

        # Clamp coordinates to be within image bounds
        px = np.clip(px, 0, tex_width - 1)
        py = np.clip(py, 0, tex_height - 1)

        # Get the color from the texture pixel data
        color = tex_pixels[py, px]
        face_colors.append(color)
    
    print(f"Sampled {len(face_colors)} face colors from texture.")
    return np.array(face_colors)