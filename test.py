import trimesh
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import json

import two_extruder_slicer as slicer

INPUT_GLB_FILE = Path("models/gemini_image_copy_7.glb")
OUTPUT_PLAN_FILE = Path("slicing_plan.json")
LAYER_HEIGHT = 0.2  # mm (Adjust based on your printer/desired quality)
FILAMENT_DIAMETER = 1.75 # mm (Used for context, not direct calculation here)

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
            
#slicer.identify_and_assign_colors_texture(mesh)
slicer.get_face_colors_from_texture(mesh)
