import cv2
import numpy as np
import dask.array as da
from numba import njit
import os
import rasterio
from rasterio.merge import merge
from multiprocessing import Pool
import psutil
import time

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
start_time = time.time()
print("Reading image metadata...")
img1_height, img1_width, img1_count, img1_crs, img1_transform, img1_dtype = get_image_info(image1_path)
img2_height, img2_width, img2_count, img2_crs, img2_transform, img2_dtype = get_image_info(image2_path)

print(f"Image 1 dimensions: {img1_height}x{img1_width}, bands: {img1_count}")
print(f"Image 2 dimensions: {img2_height}x{img2_width}, bands: {img2_count}")

# Calculate common shape (intersection)
common_height = min(img1_height, img2_height)
common_width = min(img1_width, img2_width)
print(f"Using common dimensions: {common_height}x{common_width}")

# Dynamic chunk size based on available memory (target ~50% of 5GB)
available_memory = psutil.virtual_memory().available // (1024 ** 2)  # MB
chunk_size = int((available_memory * 0.5 / 3) ** 0.5)  # 3 bands, sqrt for height/width
chunk_size = max(256, min(chunk_size, 1024))  # Clamp between 256 and 1024
print(f"Dynamic chunk size: {chunk_size}")

# Load images using rasterio with Dask
print("Loading image 1 with rasterio...")
with rasterio.open(image1_path) as src:
    img1 = da.from_array(src.read(), chunks=(img1_count, chunk_size, chunk_size)).transpose(1, 2, 0)
    img1 = img1[:common_height, :common_width, :]

print("Loading image 2 with rasterio...")
with rasterio.open(image2_path) as src:
    img2 = da.from_array(src.read(), chunks=(img2_count, chunk_size, chunk_size)).transpose(1, 2, 0)
    img2 = img2[:common_height, :common_width, :]

# Convert BGR to RGB
img1_rgb = da.stack([img1[..., 2], img1[..., 1], img1[..., 0]], axis=-1)
img2_rgb = da.stack([img2[..., 2], img2[..., 1], img2[..., 0]], axis=-1)

# Split into R, G, B channels
r1, g1, b1 = img1_rgb[..., 0], img1_rgb[..., 1], img1_rgb[..., 2]
r2, g2, b2 = img2_rgb[..., 0], img2_rgb[..., 1], img2_rgb[..., 2]

# Numba-accelerated VDVI calculation
@njit
def calculate_vdvi_chunk(r, g, b):
    denominator = 2 * g + r + b
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return (2 * g - r - b) / denominator

# Apply VDVI calculation
print("Calculating VDVI for image 1...")
vdvi1 = da.map_blocks(calculate_vdvi_chunk, r1.astype(float), g1.astype(float), b1.astype(float), dtype=float)
print("Calculating VDVI for image 2...")
vdvi2 = da.map_blocks(calculate_vdvi_chunk, r2.astype(float), g2.astype(float), b2.astype(float), dtype=float)

# Change detection and classification
print("Calculating difference and classifying changes...")
vdvi_diff = vdvi2 - vdvi1
gain_threshold = 0.2
loss_threshold = -0.2
forest_gain = (vdvi_diff > gain_threshold).astype(np.uint8) * 255
forest_loss = (vdvi_diff < loss_threshold).astype(np.uint8) * 255
no_change = ((vdvi_diff <= gain_threshold) & (vdvi_diff >= loss_threshold)).astype(np.uint8) * 128
change_map = da.stack([forest_loss, forest_gain, no_change], axis=-1)

# Multiprocessing for quantification
def count_positive_pixels(block):
    return np.sum(block > 0)

def process_array_in_blocks_mp(dask_array, block_size=chunk_size):
    shape = dask_array.shape
    with Pool(processes=8) as pool:  # Use all 8 cores
        tasks = []
        for i in range(0, shape[0], block_size):
            end_i = min(i + block_size, shape[0])
            for j in range(0, shape[1], block_size):
                end_j = min(j + block_size, shape[1])
                block = dask_array[i:end_i, j:end_j].compute()
                tasks.append(block)
        results = pool.map(count_positive_pixels, tasks)
    return sum(results)

# Quantification
pixel_area = 0.01  # 0.1m x 0.1m = 0.01 sqm per pixel
print("Calculating forest gain area...")
gain_area = process_array_in_blocks_mp(forest_gain) * pixel_area / 10000  # hectares
print("Calculating forest loss area...")
loss_area = process_array_in_blocks_mp(forest_loss) * pixel_area / 10000  # hectares

print(f"Forest Gain: {gain_area:.2f} hectares")
print(f"Forest Loss: {loss_area:.2f} hectares")

# Save change map in chunks
output_dir = "change_map_chunks"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_crs = img1_crs
output_transform = img1_transform

print("Saving change map in GeoTIFF chunks...")
chunk_files = []
for i in range(0, common_height, chunk_size):
    end_i = min(i + chunk_size, common_height)
    for j in range(0, common_width, chunk_size):
        end_j = min(j + chunk_size, common_width)
        chunk_filename = f"{output_dir}/change_map_chunk_{i}_{j}.tif"
        print(f"Processing chunk: ({i}:{end_i}, {j}:{end_j})")
        
        chunk = change_map[i:end_i, j:end_j].compute()
        chunk_transform = rasterio.transform.Affine(
            output_transform.a, output_transform.b,
            output_transform.c + j * output_transform.a,
            output_transform.d, output_transform.e,
            output_transform.f + i * output_transform.e
        )
        
        chunk_transposed = np.transpose(chunk, (2, 0, 1))  # (bands, height, width)
        with rasterio.open(
            chunk_filename, 'w', driver='GTiff', height=chunk.shape[0],
            width=chunk.shape[1], count=3, dtype=chunk.dtype,
            crs=output_crs, transform=chunk_transform
        ) as dst:
            dst.write(chunk_transposed)
        
        chunk_files.append(chunk_filename)

print(f"All chunks saved to {output_dir}/")

# Optional mosaic creation (warn about memory)
user_input = input("Do you want to create a single GeoTIFF mosaic? This may exceed 5GB RAM (y/n): ")
if user_input.lower() == 'y':
    try:
        print("Creating mosaic from chunks...")
        src_files_to_mosaic = [rasterio.open(file) for file in chunk_files]
        mosaic, out_trans = merge(src_files_to_mosaic)
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans})
        
        with rasterio.open("forest_change_map_full.tif", "w", **out_meta) as dest:
            dest.write(mosaic)
        
        for src in src_files_to_mosaic:
            src.close()
        print("Mosaic created successfully: forest_change_map_full.tif")
    except Exception as e:
        print(f"Error creating mosaic: {str(e)}")
        print("Use GDAL tools like gdal_merge.py for merging later.")
else:
    print("Mosaic not created. Use GDAL tools for merging if needed.")

print("Processing complete!")

end_time = time.time()

# Calculate the time difference
time_difference = end_time - start_time

print(f"Time taken: {time_difference} seconds")