import bpy
import os
import addon_utils

input_path = "models/birdy/birdy.glb"
output_path = "models/birdy/birdy_baked.obj"

'''
def bake_texture_to_vertex_color():
    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Import the GLB file
    try:
        bpy.ops.import_scene.gltf(filepath=input_path)
        print(f"Successfully imported {input_path}")
    except Exception as e:
        print(f"Error importing GLB file '{input_path}': {e}")
        return

    # Find the imported mesh object(s)
    # Prioritize selected objects first, then search all objects
    mesh_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not mesh_objects:
        mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']

    if not mesh_objects:
        print("No mesh objects found in the imported file or scene.")
        return

    # --- Process the first mesh object found ---
    obj = mesh_objects[0]
    print(f"Processing mesh object: {obj.name}")

    # Ensure the object is selected and active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    mesh = obj.data

    # Check for UV layers (required for texture baking)
    if not mesh.uv_layers:
        print(f"Error: Mesh '{obj.name}' has no UV layers. Cannot bake texture.")
        # Optionally, try to auto-generate UVs here if desired
        # bpy.ops.object.mode_set(mode='EDIT')
        # bpy.ops.mesh.select_all(action='SELECT')
        # bpy.ops.uv.smart_project()
        # bpy.ops.object.mode_set(mode='OBJECT')
        # if not mesh.uv_layers: return # Exit if still no UVs
        return # Exit if no UVs

    # Ensure vertex color layer exists, create if not
    if not mesh.vertex_colors:
        mesh.vertex_colors.new()
        print("Created new vertex color layer.")

    # Set the first vertex color layer as active for baking target
    mesh.vertex_colors.active_index = 0
    active_vc_layer = mesh.vertex_colors.active
    if not active_vc_layer:
         print("Error: Failed to set an active vertex color layer.")
         return

    print(f"Using vertex color layer: {active_vc_layer.name}")

    # --- Configure Baking ---
    # Switch to Cycles render engine for baking capabilities
    bpy.context.scene.render.engine = 'CYCLES'

    # Configure bake settings
    bake_settings = bpy.context.scene.render.bake
    bake_settings.target = 'VERTEX_COLORS'
    # Bake the base color (diffuse) - ignoring lighting/shading
    bake_settings.use_pass_direct = False
    bake_settings.use_pass_indirect = False
    bake_settings.use_pass_color = True # Ensure color pass is enabled for DIFFUSE type

    # --- Perform Bake ---
    print("Starting texture bake to vertex colors...")
    try:
        # Bake DIFFUSE color pass to the active vertex color layer
        bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})
        print("Bake completed successfully.")
    except Exception as e:
        print(f"Error during baking process: {e}")
        # Common issues: No active material, complex node setup, no UV map.
        return

    # --- Export OBJ ---
    # Prepare output path (same directory, _baked suffix)
    output_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, base_name + "_baked.obj")

    print(f"Exporting mesh with baked vertex colors to: {output_path}")
    try:
        bpy.ops.export_scene.obj(
            filepath=output_path,
            use_selection=True,      # Export only the selected object
            use_materials=False,     # Do not export MTL file (colors are baked)
            use_vertex_color=True,   # Crucial: Include vertex color data in OBJ
            check_existing=True,     # Overwrite existing file if needed
            axis_forward='-Z',       # Standard forward axis
            axis_up='Y'              # Standard up axis
        )
        print("Export successful.")
    except Exception as e:
        print(f"Error exporting OBJ file: {e}")

    # Optional: Clean up scene after export
    # bpy.ops.object.select_all(action='SELECT')
    # bpy.ops.object.delete(use_global=False)

# Example usage (assuming the script is run from Blender):
# Make sure to set the input_path correctly before calling the function.
# input_path = "/path/to/your/model.glb"
# bake_texture_to_vertex_color()
bake_texture_to_vertex_color()
'''

'''
def ensure_addon_enabled(addon_name):
    """Checks if an addon is enabled, and enables it if necessary."""
    enabled, loaded = addon_utils.check(addon_name)
    if not enabled:
        print(f"Add-on '{addon_name}' is not enabled. Attempting to enable...")
        try:
            addon_utils.enable(addon_name, default_set=True, persistent=False)
            # Check again after trying to enable
            enabled, loaded = addon_utils.check(addon_name)
            if enabled:
                print(f"Successfully enabled '{addon_name}' for this session.")
            else:
                print(f"ERROR: Failed to enable '{addon_name}'. Export may fail.")
                return False # Indicate failure
        except Exception as e:
            print(f"ERROR: Could not enable addon '{addon_name}': {e}")
            return False # Indicate failure
    else:
        print(f"Add-on '{addon_name}' is already enabled.") # Optional: uncomment for verbose logging
        pass
    return enabled # Return the final status

ensure_addon_enabled('io_scene_obj') # Ensure OBJ export addon is enabled

def bake_texture_to_vertex_color_and_export_obj(glb_filepath, output_obj_filepath):
    """
    Imports a GLB file, bakes the base color texture to vertex colors
    for each mesh object, and exports the scene as an OBJ file with vertex colors.

    Args:
        glb_filepath (str): Path to the input .glb file.
        output_obj_filepath (str): Path to save the output .obj file.
    """
    print(f"--- Starting Bake Process for OBJ Export ---")
    print(f"Input GLB: {glb_filepath}")
    print(f"Output OBJ: {output_obj_filepath}")
    
    # --- 1. Setup Scene ---
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # --- 2. Import GLB ---
    try:
        bpy.ops.import_scene.gltf(filepath=glb_filepath)
        print(f"Successfully imported {glb_filepath}")
    except Exception as e:
        print(f"Error importing GLB file: {e}")
        return False

    # --- 3. Prepare for Baking ---
    bpy.context.scene.render.engine = 'CYCLES'
    # Optional CPU setting:
    # bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CPU'
    # bpy.context.scene.cycles.device = 'CPU'

    # Configure Bake Settings
    bake_settings = bpy.context.scene.render.bake
    bake_settings.target = 'VERTEX_COLORS'
    cycles_bake_settings = bpy.context.scene.cycles.bake_type = 'DIFFUSE' # Simplified assignment
    bake_settings.use_pass_direct = False
    bake_settings.use_pass_indirect = False
    bake_settings.use_pass_color = True
    bake_settings.use_pass_emit = False
    # bake_settings.use_pass_ambient_occlusion = False # This attribute doesn't exist on BakeSettings

    print("Render engine set to CYCLES and bake settings configured for DIFFUSE Color to Vertex Colors.")

    vertex_color_layer_name = "BakedVertColor"

    # --- 4. Iterate and Bake Objects ---
    objects_to_process = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    if not objects_to_process:
        print("No mesh objects found in the imported scene.")
        return False

    print(f"Found {len(objects_to_process)} mesh objects to process.")

    for obj in objects_to_process:
        print(f"Processing object: {obj.name}")

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        if not obj.data or not obj.active_material:
            print(f"  Skipping {obj.name}: Missing mesh data or active material.")
            continue

        mesh = obj.data
        mat = obj.active_material

        if not mat.use_nodes:
            print(f"  Skipping {obj.name}: Material '{mat.name}' does not use nodes.")
            continue

        nodes = mat.node_tree.nodes
        principled_bsdf = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)

        if not principled_bsdf:
            print(f"  Skipping {obj.name}: No Principled BSDF node found.")
            continue

        base_color_input = principled_bsdf.inputs.get("Base Color")
        has_texture_link = False
        if base_color_input and base_color_input.is_linked:
             connected_node = base_color_input.links[0].from_node
             if connected_node.type == 'TEX_IMAGE':
                  if connected_node.image:
                      has_texture_link = True
                      print(f"  Found connected Image Texture node for Base Color.")
                  else:
                      print(f"  Warning {obj.name}: Base Color image node has no image assigned.")
             else:
                 print(f"  Info {obj.name}: Base Color input connected to non-image node ('{connected_node.bl_idname}'). Baking evaluated color.")
                 has_texture_link = True # Still attempt to bake the input color
        else:
            print(f"  Skipping {obj.name}: Principled BSDF 'Base Color' input is not linked.")
            continue

        # --- Prepare Vertex Color Layer ---
        if vertex_color_layer_name in mesh.vertex_colors:
            vc_layer = mesh.vertex_colors[vertex_color_layer_name]
        else:
            vc_layer = mesh.vertex_colors.new(name=vertex_color_layer_name)
            print(f"  Created new vertex color layer: {vertex_color_layer_name}")

        if not vc_layer:
             print(f"  Error: Could not create or find vertex color layer for {obj.name}.")
             continue

        mesh.vertex_colors.active = vc_layer
        print(f"  Set '{vertex_color_layer_name}' as active vertex color layer.")

        # --- Perform Bake ---
        print(f"  Baking material to vertex colors for {obj.name}...")
        try:
            bpy.ops.object.bake(
                type='DIFFUSE',
                pass_filter={'COLOR'},
                target='VERTEX_COLORS'
            )
            print(f"  Bake successful for {obj.name}.")

            # Optional: Remove material after baking (OBJ doesn't handle complex materials well anyway)
            # obj.active_material = None

        except RuntimeError as e:
            print(f"  Bake failed for {obj.name}: {e}")
            # continue

    # --- 5. Export Modified Scene as OBJ ---
    print(f"\nExporting scene with vertex colors to {output_obj_filepath}...")
    try:
        # Select all relevant objects for export
        bpy.ops.object.select_all(action='SELECT') # Select everything that should be exported

        bpy.ops.export_scene.obj(
            filepath=output_obj_filepath,
            check_existing=True,        # Overwrite existing file without asking
            axis_forward='-Z',          # Standard forward axis for OBJ
            axis_up='Y',                # Standard up axis for OBJ
            use_selection=False,        # Export the entire scene (since we selected all)
                                        # Change to True if you only selected specific objects
            use_animation=False,
            use_mesh_modifiers=True,    # Apply modifiers before exporting
            use_edges=True,
            use_smooth_groups=False,    # Usually not needed if normals are exported
            use_smooth_groups_bitflags=False,
            use_normals=True,           # Export normals
            use_uvs=True,               # Export UV coordinates
            use_materials=False,        # *** Don't export MTL file (we baked to vertex colors) ***
                                        # Set to True if you want an MTL based on original materials
            use_triangles=False,        # Export quads/ngons if they exist
            use_vertex_groups=False,    # Don't export vertex groups
            use_nurbs=False,
            use_colors=True,            # ***** CRITICAL: Export vertex colors *****
            global_scale=1.0,
            path_mode='AUTO'            # How paths are written (less relevant if use_materials=False)
        )
        print(f"Successfully exported scene to {output_obj_filepath}")
        return True

    except Exception as e:
        print(f"Error exporting OBJ file: {e}")
        return False

    finally:
        print("--- Bake Process Finished ---")


# --- How to Run ---
if __name__ == "__main__":
    # --- Configuration ---
    # Set the full path to your input GLB file
    input_glb = "/path/to/your/input_model.glb" # <--- CHANGE THIS

    # Set the full path for the output OBJ file
    output_obj = "/path/to/your/output_model_baked.obj" # <--- CHANGE THIS

    # --- Execute ---
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
    elif not os.path.isdir(os.path.dirname(output_path)):
         print(f"Error: Output directory not found at {os.path.dirname(output_path)}")
    else:
        bake_texture_to_vertex_color_and_export_obj(input_path, output_path)
'''


import bpy
import os

def find_base_color_image(material):
    """Finds the Image data block connected to the Base Color of a Principled BSDF."""
    if not material or not material.use_nodes:
        return None

    nodes = material.node_tree.nodes
    principled = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not principled:
        return None

    base_color_input = principled.inputs.get("Base Color")
    if not base_color_input or not base_color_input.is_linked:
        return None

    link = base_color_input.links[0]
    connected_node = link.from_node

    if connected_node.type == 'TEX_IMAGE':
        if connected_node.image:
            return connected_node.image
        else:
            print(f"  Warning: Material '{material.name}' has an Image Texture node connected to Base Color, but no image is assigned.")
            return None
    else:
        # Could try to evaluate other node types, but that's complex.
        # For this script, we only handle direct Image Texture connections.
        print(f"  Info: Material '{material.name}' Base Color is connected to a '{connected_node.bl_idname}' node, not TEX_IMAGE. Skipping direct sampling.")
        return None


def sample_texture_to_vertex_color_manual(glb_filepath, output_filepath, export_format='OBJ'):
    """
    Imports GLB, manually samples base color texture to vertex colors per loop,
    and exports the result.

    Args:
        glb_filepath (str): Path to the input .glb file.
        output_filepath (str): Path to save the output file.
        export_format (str): 'OBJ' or 'GLB'.
    """
    print(f"--- Starting Manual Texture Sampling Process ---")
    print(f"Input GLB: {glb_filepath}")
    print(f"Output {export_format}: {output_filepath}")

    # --- 0. Setup & Ensure Addons ---
    bpy.ops.wm.read_factory_settings(use_empty=True)

    if export_format.upper() == 'OBJ':
        addon_name = "io_scene_obj"
        try:
            print(f"Ensuring '{addon_name}' add-on is enabled...")
            bpy.ops.preferences.addon_enable(module=addon_name)
            print(f"'{addon_name}' add-on is enabled.")
        except Exception as e:
            print(f"Error enabling '{addon_name}': {e}. OBJ export will fail.")
            #return False
    elif export_format.upper() != 'GLB':
        print(f"Error: Unsupported export format '{export_format}'. Use 'OBJ' or 'GLB'.")
        #return False


    # --- 1. Import GLB ---
    try:
        bpy.ops.import_scene.gltf(filepath=glb_filepath)
        print(f"Successfully imported {glb_filepath}")
    except Exception as e:
        print(f"Error importing GLB file: {e}")
        return False

    # --- 2. Process Objects ---
    objects_to_process = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    if not objects_to_process:
        print("No mesh objects found in the imported scene.")
        # Decide if this is an error or just finish
        return False # Or True if exporting an empty scene is okay

    print(f"Found {len(objects_to_process)} mesh objects to process.")
    vertex_color_layer_name = "SampledTexColor"

    for obj in objects_to_process:
        print(f"\nProcessing object: {obj.name}")

        # Make object active (good practice)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        if not obj.data:
            print(f"  Skipping {obj.name}: No mesh data.")
            continue

        mesh = obj.data

        # --- Get Material and Base Color Texture ---
        if not obj.active_material:
            print(f"  Skipping {obj.name}: No active material.")
            continue
        mat = obj.active_material
        image = find_base_color_image(mat)
        print(f"mat: {mat}")
        print(f"image: {image}")

        if not image:
            print(f"  Skipping {obj.name}: Could not find a suitable Base Color Image Texture in material '{mat.name}'.")
            continue

        # --- Check if Image data is available ---
        # Important: image.pixels are only accessible if the image is loaded into memory.
        # For packed images or external images, Blender might load them on demand,
        # but `image.sample()` needs the data. Let's try to ensure it's loaded.
        if not image.has_data:
             print(f"  Image '{image.name}' has no data. Attempting to reload...")
             try:
                 image.reload()
                 if not image.has_data:
                     # Sometimes pack/unpack is needed for internal data access
                     if image.packed_file:
                         print(f"  Attempting to unpack '{image.name}' temporarily...")
                         image.unpack(method='USE_ORIGINAL') # Or 'WRITE_ORIGINAL'
                         image.reload()

             except RuntimeError as e:
                 print(f"  Error reloading/unpacking image '{image.name}': {e}")

        # Final check after trying to load
        if not image.has_data:
              print(f"  Skipping {obj.name}: Failed to load data for image '{image.name}'. Cannot sample.")
              continue
        else:
              print(f"  Image '{image.name}' ({image.size[0]}x{image.size[1]}) seems loaded.")


        # --- Get UV Layer ---
        if not mesh.uv_layers:
            print(f"  Skipping {obj.name}: Mesh has no UV layers.")
            continue
        # Use the active UV layer (usually the one used for rendering)
        uv_layer = mesh.uv_layers.active
        if not uv_layer:
            print(f"  Skipping {obj.name}: Mesh has no active UV layer.")
            continue
        print(f"  Using active UV layer: {uv_layer.name}")

        # --- Get/Create Vertex Color Layer ---
        if vertex_color_layer_name in mesh.vertex_colors:
            vc_layer = mesh.vertex_colors[vertex_color_layer_name]
            print(f"  Using existing vertex color layer: {vertex_color_layer_name}")
        else:
            vc_layer = mesh.vertex_colors.new(name=vertex_color_layer_name)
            print(f"  Created new vertex color layer: {vertex_color_layer_name}")

        if not vc_layer:
             print(f"  Error: Could not create or find vertex color layer for {obj.name}.")
             continue

        # Set the layer as active (important!)
        mesh.vertex_colors.active = vc_layer


        # --- Iterate through Loops and Sample ---
        num_loops = len(mesh.loops)
        print(f"  Sampling texture for {num_loops} loops...")

        # Access data arrays directly for potential speedup vs accessing per loop
        uv_data = uv_layer.data
        vc_data = vc_layer.data

        processed_loops = 0
        for i in range(num_loops):
            loop_uv = uv_data[i].uv # UV coordinate for this specific loop
            # loop_vert_idx = mesh.loops[i].vertex_index # Index of the vertex itself

            try:
                # Sample the image at the UV coordinate
                # sample() takes x, y in 0-1 range. It handles interpolation.
                sampled_color = image.sample(x=loop_uv[0], y=loop_uv[1])

                # Ensure color is RGBA (sample might return RGB for RGB images)
                if len(sampled_color) == 3:
                    final_color = (sampled_color[0], sampled_color[1], sampled_color[2], 1.0)
                elif len(sampled_color) == 4:
                    final_color = sampled_color
                else: # Handle grayscale or other unexpected formats
                    print(f"  Warning: Unexpected color format ({len(sampled_color)} channels) at loop {i}. Using white.")
                    final_color = (1.0, 1.0, 1.0, 1.0)

                # Assign the sampled color to the vertex color layer for this loop
                vc_data[i].color = final_color
                processed_loops += 1

            except Exception as e:
                # Catch potential errors during sampling (though less common with sample())
                print(f"  Error sampling image at loop index {i} with UV {loop_uv}: {e}")

        print(f"  Finished sampling. Assigned colors to {processed_loops}/{num_loops} loops.")

        # Optional: Remove original material if only vertex color is desired
        # obj.active_material = None

        # Deselect object after processing
        obj.select_set(False)


    # --- 3. Export Result ---
    print(f"\nExporting scene with sampled vertex colors to {output_filepath}...")
    try:
        # Select all processed objects for export (or just export the whole scene)
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects_to_process:
             # Re-select objects that were successfully processed (or just select all if preferred)
             # A simple way is just to select all mesh objects again:
             if obj.type == 'MESH':
                 obj.select_set(True)

        if export_format.upper() == 'GLB':
            bpy.ops.export_scene.gltf(
                filepath=output_filepath,
                export_format='GLB',
                use_selection=True,       # Export only selected objects
                export_colors=True,        # ***** CRITICAL: Export vertex colors *****
                export_materials='NONE' # Or 'PLACEHOLDER'/'EXPORT'
            )
        elif export_format.upper() == 'OBJ':
            bpy.ops.export_scene.obj(
                filepath=output_filepath,
                check_existing=True,
                axis_forward='-Z',
                axis_up='Y',
                use_selection=True,        # Export only selected objects
                use_mesh_modifiers=True,
                use_normals=True,
                use_uvs=True,
                use_materials=False,       # Don't export MTL file
                use_colors=True            # ***** CRITICAL: Export vertex colors *****
            )

        print(f"Successfully exported scene to {output_filepath}")
        return True

    except Exception as e:
        print(f"Error exporting {export_format} file: {e}")
        return False
    finally:
        print("--- Manual Sampling Process Finished ---")


# --- How to Run ---
if __name__ == "__main__":
    # --- Configuration ---
    input_glb = "/path/to/your/input_model.glb" # <--- CHANGE THIS
    # Choose output format and path
    output_format = "OBJ"  # Or "GLB"
    output_file = f"/path/to/your/output_model_sampled.{output_format.lower()}" # <--- CHANGE THIS

    # --- Execute ---
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
    else:
        sample_texture_to_vertex_color_manual(input_path, output_path, export_format=output_format)