# block_processor.pyx
cimport numpy as np
from cython cimport boundscheck, wraparound
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import Affine

# Since count_positive_pixels is defined in this file, no need to cimport it from elsewhere

@boundscheck(False)
@wraparound(False)
cpdef double process_array_in_blocks_cython(np.ndarray[np.uint8_t, ndim=2] arr, int block_size=1024):
    cdef int height = arr.shape[0]
    cdef int width = arr.shape[1]
    cdef double result = 0.0
    cdef int i, j, end_i, end_j
    cdef np.ndarray[np.uint8_t, ndim=2] block

    for i in range(0, height, block_size):
        end_i = min(i + block_size, height)
        for j in range(0, width, block_size):
            end_j = min(j + block_size, width)
            block = arr[i:end_i, j:end_j]
            result += count_positive_pixels(block)
    
    return result

@boundscheck(False)
@wraparound(False)
cpdef void save_chunks_cython(
    np.ndarray[np.uint8_t, ndim=3] change_map,
    str output_dir,
    int chunk_size,
    int common_height,
    int common_width,
    object output_crs,  # CRS remains a Python object
    object output_transform,  # Affine remains a Python object
    list chunk_files
):
    cdef int i, j, end_i, end_j
    cdef str chunk_filename
    cdef np.ndarray[np.uint8_t, ndim=3] chunk

    for i in range(0, common_height, chunk_size):
        end_i = min(i + chunk_size, common_height)
        for j in range(0, common_width, chunk_size):
            end_j = min(j + chunk_size, common_width)
            # Explicitly encode to bytes and decode to str for Cython compatibility
            chunk_filename = f"{output_dir}/change_map_chunk_{i}_{j}.tif".encode('utf-8').decode('utf-8')
            
            chunk = change_map[i:end_i, j:end_j]
            chunk_transform = Affine(
                output_transform.a, output_transform.b,
                output_transform.c + j * output_transform.a,
                output_transform.d, output_transform.e,
                output_transform.f + i * output_transform.e
            )
            
            with rasterio.open(
                chunk_filename, 'w',
                driver='GTiff',
                height=chunk.shape[0],
                width=chunk.shape[1],
                count=3,
                dtype=chunk.dtype,
                crs=output_crs,
                transform=chunk_transform
            ) as dst:
                dst.write(np.transpose(chunk, (2, 0, 1)))
            
            chunk_files.append(chunk_filename)

@boundscheck(False)
@wraparound(False)
cpdef void create_mosaic_cython(list chunk_files, str output_mosaic="forest_change_map_full.tif"):
    cdef list src_files_to_mosaic = []
    cdef int i, num_files = len(chunk_files)
    
    for i in range(num_files):
        src = rasterio.open(chunk_files[i])
        src_files_to_mosaic.append(src)
    
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    cdef dict out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })
    
    with rasterio.open(output_mosaic, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    for i in range(num_files):
        src_files_to_mosaic[i].close()

def create_mosaic(chunk_files, output_mosaic="forest_change_map_full.tif"):
    try:
        print("Creating mosaic from chunks...")
        create_mosaic_cython(chunk_files, output_mosaic)
        print(f"Mosaic created successfully: {output_mosaic}")
    except Exception as e:
        print(f"Error creating mosaic: {str(e)}")
        print("You can merge the chunks later using GDAL tools like gdal_merge.py")

@boundscheck(False)
@wraparound(False)
cpdef np.ndarray[np.double_t, ndim=2] calculate_vdvi_chunk(double[:, :] r, double[:, :] g, double[:, :] b):
    cdef Py_ssize_t height = r.shape[0]
    cdef Py_ssize_t width = r.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] vdvi = np.zeros((height, width), dtype=np.float64)
    cdef double denominator, numerator
    cdef Py_ssize_t i, j
    cdef double eps = 1e-10

    for i in range(height):
        for j in range(width):
            denominator = 2 * g[i, j] + r[i, j] + b[i, j]
            if denominator == 0:
                denominator = eps
            numerator = 2 * g[i, j] - r[i, j] - b[i, j]
            vdvi[i, j] = numerator / denominator

    return vdvi

@boundscheck(False)
@wraparound(False)
cpdef Py_ssize_t count_positive_pixels(np.ndarray[np.uint8_t, ndim=2] array):
    cdef Py_ssize_t height = array.shape[0]
    cdef Py_ssize_t width = array.shape[1]
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t i, j

    for i in range(height):
        for j in range(width):
            if array[i, j] > 0:
                count += 1

    return count