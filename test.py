import trimesh
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import json
import time
import test2
from PIL import Image
import pymeshlab as ml

import two_extruder_slicer as slicer

INPUT_GLB_FILE = Path("models/gemini_image_copy_7_2.glb")
OUTPUT_PLAN_FILE = Path("slicing_plan.json")
LAYER_HEIGHT = 0.2  # mm (Adjust based on your printer/desired quality)
FILAMENT_DIAMETER = 1.75 # mm (Used for context, not direct calculation here)
OUTPUT_PREFIX = "voxoutput22__"

target_color_1_rgb = [224, 157, 40] # <<< CHANGE THIS (Example: a reddish color)
target_color_2_rgb = [245, 240, 211]

VOXEL_PITCH = 0.003



def load_mesh(INPUT_GLB_FILE):
    if not INPUT_GLB_FILE.is_file():
        print(f"Error: Input file not found at {INPUT_GLB_FILE}")
        exit(1)

    # Try loading directly as a mesh, handle scenes if necessary
    try:
        # use_uv=True is important for texture coordinates if present
        # process=False prevents trimesh from merging geometries initially
        mesh = trimesh.load(INPUT_GLB_FILE, force='scene', process=False)
        print("Loaded as scene.")
    except ValueError as e:
        print(f"Warning: Could not load as scene, attempting to load as mesh. Error: {e}")
        try:
            mesh = trimesh.load(INPUT_GLB_FILE, force='mesh', process=False)
            print("Loaded as mesh.")
        except Exception as e:
            print(f"Error: Failed to load GLB file: {e}")
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
    else:
        print("Loaded object is a Trimesh mesh.")


    mesh.process()
    print("Determining color source...")
    return mesh



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




def separate_mesh_via_voxelization(mesh, face_colors, target_color_1, target_color_2, voxel_pitch):
    '''
    """Separates the mesh into two watertight meshes based on face colors using voxelization."""
    if face_colors is None or len(face_colors) != len(mesh.faces):
        print(f"Error: Invalid face_colors provided. Expected {len(mesh.faces)}, Got {len(face_colors) if face_colors is not None else 0}.")
        exit(1)

    print(f"\n--- Starting Separation (Voxel Pitch: {voxel_pitch} mm) ---")

    # 1. K-Means Clustering
    print(f"Performing K-Means clustering on {len(face_colors)} face colors...")
    try:
        # Using 'k-means++' for better initialization, increase n_init
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=15, random_state=42).fit(face_colors)
        labels = kmeans.labels_ # Array of 0s and 1s, assigning each face to a cluster
        cluster_centers = kmeans.cluster_centers_
    except Exception as e:
        print(f"Error during K-Means clustering: {e}")
        # Handle potential issue if all colors are the same
        if 'same value' in str(e).lower():
             print("KMeans failed likely due to uniform color. Cannot separate.")
             # Decide how to handle: exit, or maybe assign all to one mesh?
             exit(1)
        raise e

    print(f"Cluster centers (representative colors): {cluster_centers.astype(int).tolist()}")
    faces_cluster_0 = np.where(labels == 0)[0]
    faces_cluster_1 = np.where(labels == 1)[0]

    if len(faces_cluster_0) == 0 or len(faces_cluster_1) == 0:
        print("Error: K-Means resulted in an empty cluster. Model might be monochromatic or color detection failed.")
        exit(1)
    print(f"Cluster 0: {len(faces_cluster_0)} faces, Cluster 1: {len(faces_cluster_1)} faces")
    '''

    # --- Input Validation ---
    if face_colors is None or len(face_colors) != len(mesh.faces):
        print(f"Error: Invalid face_colors provided. Expected {mesh_faces_len}, Got {len(face_colors) if face_colors is not None else 0}.")
        return None # Indicate failure

    if len(face_colors) == 0:
        print("Error: face_colors array is empty.")
        return None

    # Ensure target colors are NumPy arrays for vectorized operations
    target_1 = np.array(target_color_1)
    target_2 = np.array(target_color_2)

    if target_1.shape != (3,) or target_2.shape != (3,):
        print(f"Error: Target colors must have shape (3,). Got {target_1.shape} and {target_2.shape}.")
        return None

    print(f"\n--- Starting Separation based on Target Colors ---")
    print(f"Target Color 1: {target_1.astype(int).tolist()}")
    print(f"Target Color 2: {target_2.astype(int).tolist()}")
    print(f"Processing {len(face_colors)} face colors...")

    # --- Distance Calculation (Vectorized) ---
    # Calculate squared Euclidean distances to avoid sqrt calculation
    # diff_1 shape: (N, 3), dist_sq_1 shape: (N,)
    diff_1 = face_colors - target_1
    dist_sq_1 = np.sum(diff_1**2, axis=1)

    # diff_2 shape: (N, 3), dist_sq_2 shape: (N,)
    diff_2 = face_colors - target_2
    dist_sq_2 = np.sum(diff_2**2, axis=1)

    # --- Assign Labels ---
    # Assign label 0 if closer to target_1 (or equidistant), label 1 if closer to target_2
    # labels will be an array of 0s and 1s.
    labels = (dist_sq_2 < dist_sq_1).astype(int)

    # --- Separate Face Indices ---
    faces_cluster_0 = np.where(labels == 0)[0]
    faces_cluster_1 = np.where(labels == 1)[0]

    # --- Post-Clustering Validation ---
    if len(faces_cluster_0) == 0 or len(faces_cluster_1) == 0:
        print("Warning: Clustering resulted in an empty cluster.")
        print(f"  - Faces closer to {target_1.astype(int).tolist()}: {len(faces_cluster_0)}")
        print(f"  - Faces closer to {target_2.astype(int).tolist()}: {len(faces_cluster_1)}")
        print("  This might happen if all mesh colors are significantly closer to one target,")
        print("  or if the target colors are poorly chosen for the input.")
        # Decide if this is an error or just a warning based on your application needs
        # exit(1) # Optional: uncomment if an empty cluster is critical error

    print(f"Cluster 0 (closer to {target_1.astype(int).tolist()}): {len(faces_cluster_0)} faces")
    print(f"Cluster 1 (closer to {target_2.astype(int).tolist()}): {len(faces_cluster_1)} faces")


    # 2. Voxelization of the *entire* mesh
    print(f"\nVoxelizing the mesh with pitch={voxel_pitch}...")
    voxel_time = time.time()
    # Use the 'fill' method to get interior voxels too, needed for solid parts
    try:
        # Note: voxelized() might require significant memory for small pitch/large mesh
        voxel_grid = mesh.voxelized(pitch=voxel_pitch).fill()
        print(f"Voxelization complete. Grid shape: {voxel_grid.shape}, Filled voxels: {voxel_grid.filled_count}")
    except Exception as e:
        print(f"Error during voxelization: {e}")
        print("Try increasing VOXEL_PITCH if this is a memory issue.")
        exit(1)

    filled_coords = voxel_grid.points # Get coordinates of filled voxel centers
    print(f"Time for voxelization: {time.time() - voxel_time:.2f} seconds")

    if len(filled_coords) == 0:
        print("Error: Voxelization resulted in no filled voxels. Check mesh scale and voxel pitch.")
        exit(1)

    # 3. Assign Colors to Voxels
    print("\nAssigning colors to voxels...")
    assign_time = time.time()
    # Find the closest face on the original mesh for each voxel center
    # This is the most computationally intensive part after voxelization itself
    try:
        print("Querying closest points on mesh surface...")
        query = trimesh.proximity.ProximityQuery(mesh)
        closest_points, distances, face_indices = query.on_surface(filled_coords)
        print(f"Finished closest point query.")
    except Exception as e:
        print(f"Error during closest point query: {e}")
        exit(1)


    # Assign voxel labels based on the label of the closest face
    voxel_labels = labels[face_indices]
    print(f"Time for color assignment: {time.time() - assign_time:.2f} seconds")

    # 4. Create Two Voxel Grids
    print("\nCreating separate voxel grids for each color cluster...")
    mask_0 = (voxel_labels == 0)
    mask_1 = (voxel_labels == 1)

    # Create boolean numpy arrays representing the two grids
    voxel_matrix_0 = np.zeros(voxel_grid.shape, dtype=bool)
    voxel_matrix_1 = np.zeros(voxel_grid.shape, dtype=bool)

    # Convert world coordinates back to grid indices
    indices_all = voxel_grid.points_to_indices(filled_coords)

    # Populate the boolean matrices
    indices_0 = indices_all[mask_0]
    indices_1 = indices_all[mask_1]

    # Use tuple indexing for efficiency
    voxel_matrix_0[tuple(indices_0.T)] = True
    voxel_matrix_1[tuple(indices_1.T)] = True

    print(f"Voxel grid 0 has {np.sum(voxel_matrix_0)} filled voxels.")
    print(f"Voxel grid 1 has {np.sum(voxel_matrix_1)} filled voxels.")

    # 5. Convert Voxel Grids to Meshes (Marching Cubes)
    print("\nRunning Marching Cubes on voxel grids...")
    marching_time = time.time()
    try:
        # Use the matrix_to_marching_cubes function for direct conversion
        # Ensure correct origin is passed (minimum corner of the voxel grid)
        # Calculate the origin of the voxel grid
        voxel_origin = voxel_grid.bounds[0]  # Lower corner of the bounding box

        mesh_0 = trimesh.voxel.ops.matrix_to_marching_cubes(
            matrix=voxel_matrix_0,
            pitch=voxel_pitch)

        mesh_1 = trimesh.voxel.ops.matrix_to_marching_cubes(
            matrix=voxel_matrix_1,
            pitch=voxel_pitch)

    except ValueError as e:
         print(f"Marching cubes failed: {e}. This might happen if a grid is empty or has isolated voxels.")
         # Check if meshes were partially created
         if 'mesh_0' not in locals() or not isinstance(mesh_0, trimesh.Trimesh) or len(mesh_0.faces)==0:
             print("Mesh 0 could not be generated.")
         if 'mesh_1' not in locals() or not isinstance(mesh_1, trimesh.Trimesh) or len(mesh_1.faces)==0:
             print("Mesh 1 could not be generated.")
         # Optionally, try exporting whichever one *did* generate, or exit
         exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during Marching Cubes: {e}")
        exit(1)


    print(f"Time for Marching Cubes: {time.time() - marching_time:.2f} seconds")

    # Validate generated meshes
    valid_mesh_0 = isinstance(mesh_0, trimesh.Trimesh) and len(mesh_0.faces) > 0
    valid_mesh_1 = isinstance(mesh_1, trimesh.Trimesh) and len(mesh_1.faces) > 0

    if not valid_mesh_0:
        print("Warning: Marching cubes for cluster 0 resulted in an empty or invalid mesh.")
    else:
        print(f"Generated Mesh 0: {len(mesh_0.vertices)} vertices, {len(mesh_0.faces)} faces. Watertight: {mesh_0.is_watertight}")
        # Optional: Post-process mesh_0 (e.g., smoothing)
        #mesh_0 = trimesh.smoothing.filter_laplacian(mesh_0, iterations=5)
        #mesh_0 = trimesh.smoothing.filter_taubin(mesh_0, iterations=10)
        #mesh_0 = trimesh.smoothing.filter_humphrey(mesh_0, alpha=0.1, beta=0.8, iterations=5)
        #mesh_0 = trimesh.smoothing.filter_mut_dif_laplacian(mesh_0, lamb=0.5, iterations=10)


    if not valid_mesh_1:
        print("Warning: Marching cubes for cluster 1 resulted in an empty or invalid mesh.")
    else:
        print(f"Generated Mesh 1: {len(mesh_1.vertices)} vertices, {len(mesh_1.faces)} faces. Watertight: {mesh_1.is_watertight}")
        # Optional: Post-process mesh_1
        #mesh_1 = trimesh.smoothing.filter_laplacian(mesh_1, iterations=5)
        #mesh_1 = trimesh.smoothing.filter_taubin(mesh_1, iterations=10)
        #mesh_1 = trimesh.smoothing.filter_humphrey(mesh_1, alpha=0.1, beta=0.8, iterations=5)
        #mesh_1 = trimesh.smoothing.filter_mut_dif_laplacian(mesh_1, lamb=0.5, iterations=10)

    if not (valid_mesh_0 or valid_mesh_1):
        print("Error: Both meshes are invalid. Cannot proceed with export.")
        return None, None
    else:
        print("Both meshes are valid. Proceeding with export.")
        return mesh_0, mesh_1
    
    





def fix_and_smooth_mesh(mesh, iterations=10, alpha=0.2, smoothing_type='laplacian'):
    """
    Fixes potential mesh issues and applies smoothing.

    Args:
        mesh (trimesh.Trimesh): The input mesh.
        iterations (int): Number of smoothing iterations.
        alpha (float): Smoothing factor for Laplacian/Humphrey.
        smoothing_type (str): 'laplacian', 'taubin', or 'humphrey'.

    Returns:
        trimesh.Trimesh: The smoothed mesh, or the original mesh if smoothing fails.
    """
    print(f"\nProcessing mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces.")
    fixed_mesh = mesh.copy()

    # Merge coplanar faces
    print("Merging coplanar faces...")
    try:
        fixed_mesh = trimesh.repair.merge_coplanar(fixed_mesh, angle_threshold=1e-2, merge_tex=True)
        print("Coplanar faces merged successfully.")
    except Exception as e:
        print(f"Error merging coplanar faces: {e}")

    # --- 1. Pre-Smoothing Fixes ---
    print("Applying pre-smoothing fixes...")
    try:
        # Remove faces with identical vertices
        print("Removing faces with identical vertices...")
        unique_faces = np.unique(fixed_mesh.faces, axis=0)
        if len(unique_faces) < len(fixed_mesh.faces):
            print(f"Removed {len(fixed_mesh.faces) - len(unique_faces)} duplicate faces.")
            fixed_mesh.update_faces(unique_faces)
        else:
            print("No duplicate faces found.")

        # Fix inconsistent winding (normals)
        print("Fixing inconsistent winding (normals)...")
        fixed_mesh.fix_normals(multibody=True) # multibody=True if mesh could have separate parts still

        # Merge close vertices to clean up the mesh
        print("Merging close vertices...")
        print(f"Number of vertices before merging: {len(fixed_mesh.vertices)}")
        try:
            fixed_mesh.merge_vertices()
            print("Close vertices merged.")
            print(f"Number of vertices after merging: {len(fixed_mesh.vertices)}")
        except Exception as e:
            print(f"Error merging close vertices: {e}")
            
        # Optional: Fix degenerate faces if they cause issues
        print("Fixing degenerate faces...")
        try:
            trimesh.repair.fix_inversion(fixed_mesh)
            trimesh.repair.fix_normals(fixed_mesh)
            print("Mesh inversion and normals fixed.")
        except Exception as e:
            print(f"Error fixing degenerate faces: {e}")

        # Final process call after potential modifications
        fixed_mesh.process(validate=True)


    except Exception as e:
        print(f"Error during pre-smoothing fixes: {e}. Proceeding without all fixes.")
        # Fallback to original mesh state for smoothing attempt if fixing failed badly
        fixed_mesh = mesh.copy()


    # --- 2. Apply Smoothing ---
    print(f"Applying {smoothing_type} smoothing with iterations={iterations}, alpha/lambda={alpha}...")
    smoothed_mesh = fixed_mesh.copy() # Work on a copy after fixes

    try:
        if smoothing_type == 'laplacian':
             trimesh.smoothing.filter_laplacian(smoothed_mesh, iterations=iterations)
        elif smoothing_type == 'taubin':
             # Taubin needs lambda and mu. Let's use alpha as lambda and a common mu.
             lambda_filter = alpha
             mu = - (lambda_filter + 0.01) # A common heuristic relation lambda = -mu
             print(f"Using Taubin with lambda={lambda_filter}, mu={mu}")
             trimesh.smoothing.filter_taubin(smoothed_mesh, iterations=iterations, lambda_filter=lambda_filter, mu=mu)
        elif smoothing_type == 'humphrey':
             trimesh.smoothing.filter_humphrey(smoothed_mesh, iterations=iterations, alpha=alpha, beta=alpha*0.5) # Beta controls shrinkage resistance
        else:
             print(f"Unknown smoothing type: {smoothing_type}. Returning un-smoothed mesh.")
             return fixed_mesh

        print("Smoothing applied successfully.")
        # Optional: Post-smoothing validation
        smoothed_mesh.process(validate=True)
        return smoothed_mesh

    except Exception as e:
        print(f"Error during {smoothing_type} smoothing: {e}. Returning mesh after pre-fixing only.")
        return fixed_mesh # Return the state after fixing but before failed smoothing




face_colors = None
mesh = load_mesh(INPUT_GLB_FILE)
face_colors = get_face_colors_from_texture(mesh)
#separate_mesh(face_colors, mesh)

start_time = time.time()
mesh_0, mesh_1 = separate_mesh_via_voxelization(mesh, face_colors, target_color_1_rgb, target_color_2_rgb, VOXEL_PITCH)

if mesh_0 is not None and mesh_1 is not None:
    #mesh_0 = fix_and_smooth_mesh(mesh_0, iterations=1, alpha=0.1, smoothing_type='laplacian')
    #mesh_1 = fix_and_smooth_mesh(mesh_1, iterations=1, alpha=0.1, smoothing_type='laplacian')

    # Convert trimesh meshes to PyMeshLab meshes
    ms_0 = ml.MeshSet()
    ms_1 = ml.MeshSet()

    # Add trimesh meshes to PyMeshLab MeshSet
    ms_0.add_mesh(ml.Mesh(mesh_0.vertices, mesh_0.faces), f"{OUTPUT_PREFIX}_0")
    ms_1.add_mesh(ml.Mesh(mesh_1.vertices, mesh_1.faces), f"{OUTPUT_PREFIX}_1")

    print("Converted trimesh meshes to PyMeshLab meshes.")

    #ms_0.apply_coord_laplacian_smoothing(stepsmoothnum=3, boundary=True, cotangentweight=True)
    #ms_1.apply_coord_laplacian_smoothing(stepsmoothnum=3, boundary=True, cotangentweight=True)
    print("Applied Laplacian smoothing to PyMeshLab meshes.")

    output_stl_0 = f"{OUTPUT_PREFIX}_0.stl"
    output_stl_1 = f"{OUTPUT_PREFIX}_1.stl"

    '''
    try:
        print(f"\nExporting {output_stl_0}...")
        mesh_0.export(output_stl_0)
        print(f"Exporting {output_stl_1}...")
        mesh_1.export(output_stl_1)
        print("Export complete.")
    except Exception as e:
        print(f"Error exporting STL files: {e}")
    '''

    # Save the meshes using PyMeshLab
    try:
        ms_0.save_current_mesh(output_stl_0)
        ms_1.save_current_mesh(output_stl_1)
        print(f"Exported {output_stl_0} and {output_stl_1} successfully.")
    except Exception as e:
        print(f"Error exporting STL files: {e}")

    total_time = time.time() - start_time
    print(f"\n--- Separation Finished ---")
    print(f"Total time: {total_time:.2f} seconds")

#test2.separate_mesh(face_colors, mesh)

