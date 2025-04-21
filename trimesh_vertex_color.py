import trimesh
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans
import random


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
        if isinstance(mesh, trimesh.Scene) and mesh.geometry:
            mesh = list(mesh.geometry.values())[0] # Take the first mesh found
            print(f"Extracted mesh: {mesh}")
            print("Proceeding with the first mesh found in the scene.")
            if not isinstance(mesh, trimesh.Trimesh):
                print("Error: Could not extract a valid Trimesh object from the scene.")
                exit(1)
        else:
            print("Error: Cannot proceed.")
            exit(1)
            
    except ValueError as e:
        print(f"Warning: Could not load as scene, attempting to load as mesh. Error: {e}")
        try:
            mesh = trimesh.load(INPUT_GLB_FILE, force='mesh', process=False)
            print("Loaded as mesh.")
        except Exception as e:
            print(f"Error: Failed to load GLB file: {e}")
            exit(1)
            

    mesh.process()
    return mesh

def get_colors_from_texture(mesh):
    """
    Samples the texture color for each face based on UV coordinates.
    Assumes TextureVisuals and a baseColorTexture.
    Returns a numpy array of RGB colors, one per face.
    """
    # -------- Checking for texture source --------
    #############################################
    print("Determining color source...")
    if not isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        print("Warning: Mesh visual is not TextureVisuals.")
        raise TypeError("Mesh visual is not TextureVisuals.")
    else:
        print("Mesh visual is Texture.")

    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        print("Warning: Mesh does not have UV coordinates.")
        raise ValueError("Mesh does not have UV coordinates.")
    else:
        print("Mesh has UV coordinates.")

    if not hasattr(mesh.visual, 'material') or not hasattr(mesh.visual.material, 'baseColorTexture') or mesh.visual.material.baseColorTexture is None:
        print("Warning: Mesh material does not have a baseColorTexture.")
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
                print("Extracting texture from PBR metallic roughness.")
                tex_info = mesh.visual.material.pbrMetallicRoughness.baseColorTexture['source']
                # Need to find the image data in the gltf resources
                # This is complex and depends on the GLTF structure trimesh exposes
                # A simpler trimesh load might directly give a PIL image.
                # Let's assume mesh.visual.material.baseColorTexture IS a PIL Image for now.
                # If not, this part needs significant GLTF structure diving.
                raise NotImplementedError("Texture loading from GLTF structure not fully implemented here.")
            else:
                print("Cannot interpret texture as PIL Image.")
                raise ValueError("Cannot access texture image data.")
        except Exception as e:
            print(f"Error: Could not interpret texture. Error: {e}")
            raise ValueError(f"Could not interpret texture: {e}")
    else:
        print("Texture is a PIL Image.")
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

    # -------- Extracting texture pixels --------
    #############################################
    tex_pixels = np.array(texture)
    tex_height, tex_width, _ = tex_pixels.shape
    uv = mesh.visual.uv
    # Reshape texture pixels for clustering
    reshaped_pixels = tex_pixels.reshape(-1, 3)

    # Cluster colors into two

    '''
    # Map UV coordinates to pixel indices
    uv[:, 0] = uv[:, 0] * (tex_width - 1)
    uv[:, 1] = (1 - uv[:, 1]) * (tex_height - 1)  # Flip Y-axis for image coordinates
    uv = np.round(uv).astype(int)
    '''

    # Cluster colors into two groups
    kmeans = KMeans(n_clusters=2, random_state=0).fit(reshaped_pixels)
    cluster_centers = kmeans.cluster_centers_.astype(int)  # Get the average color of each cluster
    labels = kmeans.labels_  # Get the cluster label for each pixel

    # Replace pixel colors with their corresponding cluster center
    posterized_pixels = np.array([cluster_centers[label] for label in labels])
    posterized_pixels = posterized_pixels.reshape(tex_height, tex_width, 3)

    # Create a new PIL image from the posterized pixels
    posterized_texture = Image.fromarray(posterized_pixels.astype('uint8'), 'RGB')
    posterized_texture.show()
    # Print the RGB values of the two posterized colors
    print("Posterized colors (RGB):")
    for i, color in enumerate(cluster_centers):
        print(f"Color {i + 1}: {color}")
    
    def sample_points_on_face(vertices, uv_coords, num_samples=5):
        """
        Randomly sample points on a triangle face and their corresponding UVs.
        Returns barycentric coordinates, 3D points, and UVs.
        """
        samples = []
        for _ in range(num_samples):
            # Random barycentric coordinates
            r1 = np.sqrt(random.random())
            r2 = random.random()
            a = 1 - r1
            b = r1 * (1 - r2)
            c = r1 * r2
            bary = np.array([a, b, c])
            # 3D point
            point = a * vertices[0] + b * vertices[1] + c * vertices[2]
            # UV point
            uv = a * uv_coords[0] + b * uv_coords[1] + c * uv_coords[2]
            samples.append((bary, point, uv))
        return samples

    def get_posterized_color(uv, posterized_pixels, tex_width, tex_height):
        u = np.clip(uv[0], 0, 1)
        v = np.clip(uv[1], 0, 1)
        x = int(round(u * (tex_width - 1)))
        y = int(round((1 - v) * (tex_height - 1)))
        #print(f"UV: ({u:.3f}, {v:.3f}) -> RGB: {posterized_pixels[y, x]}")
        return tuple(posterized_pixels[y, x])

    def face_sampling_and_subdivide(vertices, uvs, posterized_pixels, cluster_centers, tex_width, tex_height, max_depth=5, num_samples=5, depth=0):
        """
        Recursively sample and subdivide a triangle face until all samples fall into one posterized color.
        Returns a list of faces (each as (vertices, uvs)) that are homogeneous.
        """
        samples = sample_points_on_face(vertices, uvs, num_samples)
        colors = [get_posterized_color(uv, posterized_pixels, tex_width, tex_height) for _, _, uv in samples]
        # Which cluster does each color belong to?
        color_labels = []
        for color in colors:
            dists = [np.linalg.norm(np.array(color) - np.array(center)) for center in cluster_centers]
            color_labels.append(np.argmin(dists))
        if len(set(color_labels)) == 1 or depth >= max_depth:
            # Homogeneous or max depth reached
            #print(f"Homogeneous face at depth {depth}: {color_labels[0]}")
            return [(vertices, uvs, color_labels[0])]
        else:
            # Subdivide into 4 smaller triangles (midpoint subdivision)
            v0, v1, v2 = vertices
            uv0, uv1, uv2 = uvs
            vm01 = (v0 + v1) / 2
            vm12 = (v1 + v2) / 2
            vm20 = (v2 + v0) / 2
            uvm01 = (uv0 + uv1) / 2
            uvm12 = (uv1 + uv2) / 2
            uvm20 = (uv2 + uv0) / 2
            # 4 new triangles
            tris = [
                ([v0, vm01, vm20], [uv0, uvm01, uvm20]),
                ([vm01, v1, vm12], [uvm01, uv1, uvm12]),
                ([vm20, vm12, v2], [uvm20, uvm12, uv2]),
                ([vm01, vm12, vm20], [uvm01, uvm12, uvm20]),
            ]
            result = []
            for verts, uvs_ in tris:
                result.extend(face_sampling_and_subdivide(
                    np.array(verts), np.array(uvs_), posterized_pixels, cluster_centers,
                    tex_width, tex_height, max_depth, num_samples, depth + 1
                ))
            # print(f"Subdivided face at depth {depth}: {color_labels}")
            # Return the subdivided faces
            return result

    # --- Main logic: Loop through all faces and subdivide as needed ---
    faces = mesh.faces
    vertices = mesh.vertices

    all_faces = []
    for i, face in enumerate(faces):
        face_verts = vertices[face]
        face_uvs = uv[face]
        #print(f"Processing face {i + 1}/{len(faces)}")
        homogeneous_faces = face_sampling_and_subdivide(
            face_verts, face_uvs, posterized_pixels, cluster_centers, tex_width, tex_height
        )
        all_faces.extend(homogeneous_faces)

    print(f"Total subdivided faces: {len(all_faces)}")
    # Optionally, you can now use all_faces for further processing or export.

    # --- Collect unique vertices and assign colors ---
    vertex_map = {}  # (x, y, z, r, g, b) -> new index
    vertex_list = []  # list of (x, y, z, r, g, b)
    face_indices = []  # list of (i0, i1, i2)

    for verts, uvs, color_label in all_faces:
        idxs = []
        for v, uv in zip(verts, uvs):
            # Get posterized color at this UV
            color = get_posterized_color(uv, posterized_pixels, tex_width, tex_height)
            # Normalize RGB to [0, 1] for OBJ
            r, g, b = [c / 255.0 for c in color]
            key = (round(v[0], 6), round(v[1], 6), round(v[2], 6), round(r, 6), round(g, 6), round(b, 6))
            if key not in vertex_map:
                vertex_map[key] = len(vertex_list) + 1  # OBJ indices start at 1
                vertex_list.append((v[0], v[1], v[2], r, g, b))
            idxs.append(vertex_map[key])
        face_indices.append(tuple(idxs))

    # --- Export to OBJ ---
    output_obj_file = "models/birdy/birdy_posterized.obj"
    with open(output_obj_file, 'w') as f:
        for v in vertex_list:
            f.write(f"v {v[0]} {v[1]} {v[2]} {v[3]} {v[4]} {v[5]}\n")
        for face in face_indices:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    print(f"Exported subdivided mesh with posterized vertex colors to {output_obj_file}")
    
    '''
    # Get RGB values for each vertex
    vertex_colors = tex_pixels[uv[:, 1], uv[:, 0], :3]  # Extract RGB values

    # Cluster colors into two groups
    kmeans = KMeans(n_clusters=2, random_state=0).fit(vertex_colors)
    cluster_centers = kmeans.cluster_centers_  # Get the average color of each cluster
    labels = kmeans.labels_  # Get the cluster label for each vertex

    # Replace vertex colors with their corresponding cluster average
    clustered_colors = np.array([cluster_centers[label] for label in labels])

    # Add clustered RGB values to vertices
    vertices_with_colors = np.hstack((mesh.vertices, clustered_colors / 255.0))  # Normalize RGB to [0, 1]
    

    # -------- Exporting to OBJ --------
    #############################################
    # Write to OBJ file
    output_obj_file = "models/birdy/birdy_cv.obj"
    with open(output_obj_file, 'w') as f:
        for vertex in vertices_with_colors:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]} {vertex[3]} {vertex[4]} {vertex[5]}\n")
        for face in mesh.faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    print(f"Exported mesh with clustered vertex colors to {output_obj_file}")
    '''



INPUT_GLB_FILE = Path("models/birdy/birdy.glb")
mesh = load_mesh(INPUT_GLB_FILE)
get_colors_from_texture(mesh)