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
    else:
        print("Mesh visual is Texture.")
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
         raise ValueError("Mesh does not have UV coordinates.")
    else:
        print("Mesh has UV coordinates.")
    if not hasattr(mesh.visual, 'material') or \
       not hasattr(mesh.visual.material, 'baseColorTexture') or \
       mesh.visual.material.baseColorTexture is None:
        raise ValueError("Mesh material does not have a baseColorTexture.")
    else:
        print("Mesh material has baseColorTexture.")

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
    else:
        print("Texture is a valid PIL Image.")

    # Ensure texture is RGB
    if texture.mode == 'RGBA':
        print("Converting RGBA texture to RGB.")
        # Create a white background image
        bg = Image.new('RGB', texture.size, (255, 255, 255))
        # Paste the RGBA image onto the white background
        bg.paste(texture, (0, 0), texture)
        texture = bg
    elif texture.mode != 'RGB':
        print(f"Warning: Texture mode is not RGB (mode: {texture.mode}).")
        texture = texture.convert('RGB')
    else:
        print("Texture is already RGB.")

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


def get_face_colors_from_vertex_colors(mesh):
    """
    Calculates average color for each face based on vertex colors.
    Assumes VertexColorVisuals.
    Returns a numpy array of RGB colors, one per face.
    """
    if not isinstance(mesh.visual, trimesh.visual.color.VertexColorVisuals):
         raise TypeError("Mesh visual is not VertexColorVisuals.")
    if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
        raise ValueError("Mesh does not have vertex colors.")

    vertex_colors = mesh.visual.vertex_colors[:, :3] # Use only RGB, ignore alpha if present
    face_colors = []

    for face in mesh.faces:
        # Get colors of vertices for the face and calculate the mean
        face_color_avg = vertex_colors[face].mean(axis=0)
        face_colors.append(face_color_avg)

    return np.array(face_colors)

def get_face_colors_from_face_colors(mesh):
    """
    Gets colors directly from face colors.
    Assumes FaceColorVisuals.
    Returns a numpy array of RGB colors, one per face.
    """
    if not isinstance(mesh.visual, trimesh.visual.color.FaceColorVisuals):
         raise TypeError("Mesh visual is not FaceColorVisuals.")
    if not hasattr(mesh.visual, 'face_colors') or mesh.visual.face_colors is None:
        raise ValueError("Mesh does not have face colors.")

    # Use only RGB, ignore alpha if present
    return mesh.visual.face_colors[:, :3]


def separate_mesh_by_color(glb_path, output_prefix):
    """
    Loads a GLB, determines face colors, clusters them into 2 groups,
    and saves two separate STL files.
    """
    print(f"Loading mesh from {glb_path}...")
    # Try loading directly as a mesh, handle scenes if necessary
    try:
        # use_uv=True is important for texture coordinates if present
        # process=False prevents trimesh from merging geometries initially
        loaded_data = trimesh.load(glb_path, force='scene', process=False)
    except ValueError as e:
        print(f"Warning: Could not load as scene, attempting to load as mesh. Error: {e}")
        try:
            loaded_data = trimesh.load(glb_path, force='mesh', process=False)
        except Exception as e:
            print(f"Error: Failed to load GLB file: {e}")
            return

    if isinstance(loaded_data, trimesh.Scene):
        # Combine all meshes in the scene into one
        # Note: This assumes you WANT to combine them. If they are separate
        # parts meant to be processed individually, you'd loop through
        # loaded_data.geometry.values() instead.
        print("Scene detected, concatenating geometries...")
        if not loaded_data.geometry:
             print("Error: Scene contains no geometry.")
             return
        mesh = trimesh.util.concatenate(list(loaded_data.geometry.values()))
    elif isinstance(loaded_data, trimesh.Trimesh):
        mesh = loaded_data
    else:
        print(f"Error: Loaded data is not a Trimesh or Scene ({type(loaded_data)}).")
        return

    # Ensure the mesh has faces
    if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
        print("Error: Mesh has no faces.")
        return

    # Process the mesh to ensure standard format (e.g., unwelding vertices if needed for texture/color splits)
    # This can potentially change vertex/face count and order
    mesh.process()

    print("Determining color source...")
    face_colors = None
    try:
        if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals) and \
           hasattr(mesh.visual.material, 'baseColorTexture') and \
           mesh.visual.material.baseColorTexture is not None:
            print("Color source: Texture mapping")
            face_colors = get_face_colors_from_texture(mesh)

        elif isinstance(mesh.visual, trimesh.visual.color.VertexColorVisuals) and \
             hasattr(mesh.visual, 'vertex_colors') and \
             mesh.visual.vertex_colors is not None:
            print("Color source: Vertex colors")
            face_colors = get_face_colors_from_vertex_colors(mesh)

        elif isinstance(mesh.visual, trimesh.visual.color.FaceColorVisuals) and \
             hasattr(mesh.visual, 'face_colors') and \
             mesh.visual.face_colors is not None:
             print("Color source: Face colors")
             face_colors = get_face_colors_from_face_colors(mesh)

        else:
            # Check for simple PBR base color factor - this wouldn't usually be "two colors"
            # but handle as a fallback maybe? Or raise error.
            if hasattr(mesh.visual, 'material') and \
               hasattr(mesh.visual.material, 'pbrMetallicRoughness') and \
               hasattr(mesh.visual.material.pbrMetallicRoughness, 'baseColorFactor'):
                 base_color = mesh.visual.material.pbrMetallicRoughness.baseColorFactor[:3]
                 print(f"Color source: Simple material baseColorFactor {base_color}. Cannot split by two colors.")
                 # Assign all faces the same color - splitting won't work well
                 # face_colors = np.tile(np.array(base_color) * 255, (len(mesh.faces), 1))
                 print("Error: Model seems to have only one uniform color. Cannot separate into two parts based on color.")
                 return
            else:
                 print("Error: Could not determine valid color source (Texture, Vertex Colors, or Face Colors).")
                 return

    except (TypeError, ValueError, NotImplementedError, AttributeError) as e:
        print(f"Error processing color information: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during color processing: {e}")
        import traceback
        traceback.print_exc()
        return


    if face_colors is None or len(face_colors) != len(mesh.faces):
        print(f"Error: Failed to get valid color for each face. Expected {len(mesh.faces)}, Got {len(face_colors) if face_colors is not None else 0}.")
        return

    print(f"Performing K-Means clustering on {len(face_colors)} face colors...")
    # Use K-Means to find the 2 dominant colors
    try:
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(face_colors)
        labels = kmeans.labels_ # Array of 0s and 1s, assigning each face to a cluster
        cluster_centers = kmeans.cluster_centers_ # The two representative colors
    except Exception as e:
        print(f"Error during K-Means clustering: {e}")
        return

    print(f"Cluster centers (representative colors): {cluster_centers.astype(int).tolist()}")

    # Get the indices of the faces for each cluster
    faces_cluster_0 = np.where(labels == 0)[0]
    faces_cluster_1 = np.where(labels == 1)[0]

    if len(faces_cluster_0) == 0 or len(faces_cluster_1) == 0:
        print("Error: K-Means resulted in an empty cluster. The model might be monochromatic or have insufficient color variation.")
        return

    print(f"Splitting mesh: {len(faces_cluster_0)} faces in cluster 0, {len(faces_cluster_1)} faces in cluster 1")

    # Create new meshes using trimesh's submesh capability
    # 'append=True' ensures vertices are copied, making the meshes independent
    try:
        mesh_0 = mesh.submesh([faces_cluster_0], append=True)
        mesh_1 = mesh.submesh([faces_cluster_1], append=True)
    except Exception as e:
        # Sometimes submesh fails if faces are degenerate after processing
        print(f"Error creating submeshes: {e}")
        print("Attempting alternative splitting method (masking)...")
        try:
            mask_0 = np.zeros(len(mesh.faces), dtype=bool)
            mask_0[faces_cluster_0] = True
            mesh_0 = mesh.copy()
            mesh_0.update_faces(mask_0)
            mesh_0.remove_unreferenced_vertices()

            mask_1 = np.zeros(len(mesh.faces), dtype=bool)
            mask_1[faces_cluster_1] = True
            mesh_1 = mesh.copy()
            mesh_1.update_faces(mask_1)
            mesh_1.remove_unreferenced_vertices()
            # This masking method might be less robust for clean geometry separation for printing
            print("Alternative splitting method finished. Check outputs carefully.")
        except Exception as e2:
            print(f"Alternative splitting method also failed: {e2}")
            return


    # Export the meshes as STL files
    output_stl_0 = f"{output_prefix}_color_0.stl"
    output_stl_1 = f"{output_prefix}_color_1.stl"

    try:
        print(f"Exporting {output_stl_0}...")
        mesh_0.export(output_stl_0)
        print(f"Exporting {output_stl_1}...")
        mesh_1.export(output_stl_1)
        print("Export complete.")
    except Exception as e:
        print(f"Error exporting STL files: {e}")