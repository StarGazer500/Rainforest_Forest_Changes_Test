import cv2
import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
import os
import rasterio
from rasterio.merge import merge
import time

# Import the Cython module
from cython_optimized_funcs import calculate_vdvi_chunk,process_array_in_blocks_cython,save_chunks_cython,create_mosaic_cython
# from block_processor import process_array_in_blocks_cython,save_chunks_cython

# File paths
image1_path = "/media/martin/LINUX MINT/Sub_comp50a_ECW/2023.tif"  # Older image
image2_path = "/media/martin/LINUX MINT/Sub_comp50a_ECW/19-12-2024.tif"  # Newer image

# Get image shapes and geospatial metadata
def get_image_info(file_path):
    with rasterio.open(file_path) as src:
        height = src.height
        width = src.width
        count = src.count
        crs = src.crs
        transform = src.transform
        dtype = src.dtypes[0]
    return height, width, count, crs, transform, dtype

# Get image info from both images
start_time = time.time()
print("Reading image metadata...")
img1_height, img1_width, img1_count, img1_crs, img1_transform, img1_dtype = get_image_info(image1_path)
img2_height, img2_width, img2_count, img2_crs, img2_transform, img2_dtype = get_image_info(image2_path)

print(f"Image 1 dimensions: {img1_height}x{img1_width}, bands: {img1_count}")
print(f"Image 2 dimensions: {img2_height}x{img2_width}, bands: {img2_count}")

# Calculate the common shape
common_height = min(img1_height, img2_height)
common_width = min(img1_width, img2_width)
print(f"Using common dimensions: {common_height}x{common_width}")

# Define chunk size for processing
chunk_size = 1024  # Process in 1024x1024 chunks

# Load images using Dask
print("Loading image 1...")
img1 = da.from_array(cv2.imread(image1_path), chunks=(chunk_size, chunk_size, 3))
img1 = img1[:common_height, :common_width, :]

print("Loading image 2...")
img2 = da.from_array(cv2.imread(image2_path), chunks=(chunk_size, chunk_size, 3))
img2 = img2[:common_height, :common_width, :]

# Verify shapes
print(f"Processed Image 1 shape: {img1.shape}")
print(f"Processed Image 2 shape: {img2.shape}")

# Convert BGR to RGB
img1_rgb = da.stack([img1[..., 2], img1[..., 1], img1[..., 0]], axis=-1)
img2_rgb = da.stack([img2[..., 2], img2[..., 1], img2[..., 0]], axis=-1)

# Split into R, G, B channels
r1, g1, b1 = img1_rgb[..., 0], img1_rgb[..., 1], img1_rgb[..., 2]
r2, g2, b2 = img2_rgb[..., 0], img2_rgb[..., 1], img2_rgb[..., 2]

# Apply VDVI calculation using Cython via Dask
print("Calculating VDVI for image 1...")
vdvi1 = da.map_blocks(calculate_vdvi_chunk, r1.astype(float), g1.astype(float), b1.astype(float), dtype=float)
print("Calculating VDVI for image 2...")
vdvi2 = da.map_blocks(calculate_vdvi_chunk, r2.astype(float), g2.astype(float), b2.astype(float), dtype=float)

# Change detection
print("Calculating difference...")
vdvi_diff = vdvi2 - vdvi1

# Thresholds
gain_threshold = 0.2
loss_threshold = -0.2

# Classify changes
print("Classifying changes...")
forest_gain = (vdvi_diff > gain_threshold).astype(np.uint8) * 255
forest_loss = (vdvi_diff < loss_threshold).astype(np.uint8) * 255
no_change = ((vdvi_diff <= gain_threshold) & (vdvi_diff >= loss_threshold)).astype(np.uint8) * 128

# Create change map (R: loss, G: gain, B: no change)
change_map = da.stack([forest_loss, forest_gain, no_change], axis=-1)

# Process in blocks to avoid memory issues
# def process_array_in_blocks(dask_array, block_size=1024):
#     """Process a dask array in blocks using the Cython count_positive_pixels function"""
#     shape = dask_array.shape
#     result = 0
    
#     for i in range(0, shape[0], block_size):
#         end_i = min(i + block_size, shape[0])
#         for j in range(0, shape[1], block_size):
#             end_j = min(j + block_size, shape[1])
            
#             print(f"Processing block: ({i}:{end_i}, {j}:{end_j})")
#             block = dask_array[i:end_i, j:end_j].compute()
#             result += count_positive_pixels(block)
    
#     return result

# Quantification
pixel_area = 0.01  # 0.1m x 0.1m = 0.01 sqm per pixel

print("Calculating forest gain area...")
forest_gain_computed = forest_gain.compute()  # Compute once
gain_area = process_array_in_blocks_cython(forest_gain_computed) * pixel_area / 10000  # hectares

print("Calculating forest loss area...")
forest_loss_computed = forest_loss.compute()  # Compute once
loss_area = process_array_in_blocks_cython(forest_loss_computed) * pixel_area / 10000  # hectares

# print("Calculating forest gain area...")
# gain_area = process_array_in_blocks(forest_gain) * pixel_area / 10000  # hectares

# print("Calculating forest loss area...")
# loss_area = process_array_in_blocks(forest_loss) * pixel_area / 10000  # hectares

print(f"Forest Gain: {gain_area:.2f} hectares")
print(f"Forest Loss: {loss_area:.2f} hectares")

# Create directory for output chunks
output_dir = "change_map_chunks"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Use the geospatial metadata from the first image
output_crs = img1_crs
output_transform = img1_transform

# Save change map in GeoTIFF chunks
print("Saving change map in GeoTIFF chunks...")
chunk_files = []

change_map_computed = change_map.compute()  # Compute once
save_chunks_cython(change_map_computed, output_dir, chunk_size, common_height, common_width, output_crs, output_transform, chunk_files)

# for i in range(0, common_height, chunk_size):
#     end_i = min(i + chunk_size, common_height)
#     for j in range(0, common_width, chunk_size):
#         end_j = min(j + chunk_size, common_width)
        
#         chunk_filename = f"{output_dir}/change_map_chunk_{i}_{j}.tif"
#         print(f"Processing chunk: ({i}:{end_i}, {j}:{end_j})")
        
#         # Compute this chunk of the change map
#         chunk = change_map[i:end_i, j:end_j].compute()
        
#         # Calculate the updated transform for this chunk
#         from rasterio.transform import Affine
#         chunk_transform = Affine(
#             output_transform.a, output_transform.b,
#             output_transform.c + j * output_transform.a,
#             output_transform.d, output_transform.e,
#             output_transform.f + i * output_transform.e
#         )
        
#         # Convert shape for rasterio
#         chunk_transposed = np.transpose(chunk, (2, 0, 1))
        
#         # Write the chunk as a GeoTIFF
#         with rasterio.open(
#             chunk_filename,
#             'w',
#             driver='GTiff',
#             height=chunk.shape[0],
#             width=chunk.shape[1],
#             count=3,
#             dtype=chunk.dtype,
#             crs=output_crs,
#             transform=chunk_transform,
#         ) as dst:
#             dst.write(chunk_transposed)
        
#         chunk_files.append(chunk_filename)
        
#         print(f"Saved chunk: {chunk_filename}")

# print(f"All chunks saved to {output_dir}/")

# Optionally create a mosaic
user_input = input("Do you want to create a single GeoTIFF mosaic from all chunks? This may require significant memory (y/n): ")
if user_input.lower() == 'y':
    try:
        print("Creating mosaic from chunks...")
        
        create_mosaic_cython(chunk_files)
            
        print(f"Mosaic created successfully:")
    except Exception as e:
        print(f"Error creating mosaic: {str(e)}")
        print("You can merge the chunks later using GDAL tools like gdal_merge.py")
else:
    print("Mosaic not created. You can merge the chunks later using GDAL tools.")

print("Processing complete!")

end_time = time.time()
time_difference = end_time - start_time
print(f"Time taken: {time_difference} seconds")