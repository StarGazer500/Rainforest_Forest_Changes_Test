import rasterio
import numpy as np
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns
import streamlit as st
import cv2
import tempfile
import os
from contextlib import contextmanager
from rasterio.enums import Resampling
import asyncio
import logging
import gc
import time
import tifffile

# @contextmanager
def safe_file_handle(uploaded_file):
    """
    Safely handle uploaded files by saving them to temporary files.
    """
    if uploaded_file is None:
        yield None
        return
        
    try:
        # Create a temporary file
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            # Write contents to temporary file
            tmp_file.write(uploaded_file.getvalue())
            tmp_file.flush()
            yield tmp_file.name
    finally:
        # Clean up the temporary file
        if 'tmp_file' in locals():
            os.unlink(tmp_file.name)

# @st.cache_data
def threshold_ridler_calvard(image, max_iter=100, eps=1e-5):
    """
    Ridler-Calvard thresholding implementation
    """
    t = np.mean(image)
    for i in range(max_iter):
        foreground = image[image >= t]
        background = image[image < t]
        mean_fg = np.mean(foreground) if len(foreground) > 0 else t
        mean_bg = np.mean(background) if len(background) > 0 else t
        new_t = (mean_fg + mean_bg) / 2
        if abs(t - new_t) < eps:
            return new_t
        t = new_t
    return t

async def process_in_batches(file_path, reference_shape=None, batch_size=256):
    """
    Process an image in batches with proper edge handling for both source and target dimensions.
    """
    try:
        with rasterio.Env():
            with rasterio.open(file_path) as src:
                nodata = src.nodata
                dtype = np.float32
                
                # Get source dimensions
                src_shape = (src.height, src.width)
                print(f"Source image shape: {src_shape}")
                
                # Set output shape
                out_shape = reference_shape or src_shape
                print(f"Output shape: {out_shape}")
                
                # Prepare output array
                image = np.zeros(out_shape, dtype=dtype)
                
                # Loop through the image in batches
                for i in range(0, out_shape[0], batch_size):
                    for j in range(0, out_shape[1], batch_size):
                        # Calculate target dimensions for this batch
                        target_height = min(batch_size, out_shape[0] - i)
                        target_width = min(batch_size, out_shape[1] - j)
                        
                        # Calculate source dimensions for this batch
                        source_height = min(batch_size, src.height - i) if i < src.height else 0
                        source_width = min(batch_size, src.width - j) if j < src.width else 0
                        
                        print(f"Processing batch at position ({i}, {j})")
                        print(f"Target dimensions: {target_height}x{target_width}")
                        print(f"Source dimensions: {source_height}x{source_width}")
                        
                        # Skip if we're completely outside source bounds
                        if source_height <= 0 or source_width <= 0:
                            image[i:i+target_height, j:j+target_width] = 0
                            continue
                        
                        try:
                            # Read from source
                            window = rasterio.windows.Window(
                                col_off=j,
                                row_off=i,
                                width=source_width,
                                height=source_height
                            )
                            
                            batch = src.read(1, window=window)
                            print(f"Read batch shape: {batch.shape}")
                            
                            # Handle nodata values
                            if nodata is not None:
                                batch = np.where(batch == nodata, 0, batch)
                            
                            # Convert to float32
                            batch = batch.astype(dtype)
                            
                            # Process non-zero values
                            valid_pixels = batch > 0
                            if np.any(valid_pixels):
                                min_val = np.percentile(batch[valid_pixels], 2)
                                max_val = np.percentile(batch[valid_pixels], 98)
                                
                                if max_val > min_val:
                                    batch = (batch - min_val) / (max_val - min_val)
                                    batch = np.clip(batch, 0, 1)
                            
                            # Create target batch (filled with zeros)
                            target_batch = np.zeros((target_height, target_width), dtype=dtype)
                            
                            # Copy data into target batch
                            copy_height = min(source_height, target_height)
                            copy_width = min(source_width, target_width)
                            target_batch[:copy_height, :copy_width] = batch[:copy_height, :copy_width]
                            
                            # Assign to output array
                            image[i:i+target_height, j:j+target_width] = target_batch
                        
                            
                        except Exception as batch_error:
                            print(f"Error processing batch at ({i}, {j}): {str(batch_error)}")
                            print(f"Batch dimensions: source={source_height}x{source_width}, target={target_height}x{target_width}")
                            raise
                    # Force memory cleanup after each batch
                    # gc.collect()
                # Force memory cleanup before return
                # gc.collect()
                return image, src.profile
                
    except rasterio.errors.RasterioIOError as rio_err:
        print(f"Rasterio IO Error: {str(rio_err)}")
        return None, None
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, None
 
async def detect_changes_multi_method(image1, image2, reference_shape=None, batch_size=256):
    try:
        # Input validation
        if image1.shape != image2.shape:
            raise ValueError("Input images must have the same dimensions")
        
        # Determine the reference shape based on the first image
        if reference_shape is None:
            reference_shape = image1.shape
        
        diff_shape = reference_shape
        results = {}
        
        # Initialize result images
        new = np.zeros_like(image1)
        abs_changes = new
        stat_changes = new
        perc_changes = new
        otsu_changes = new

        print("Starting batch processing...")

        # Process the images in batches
        for i in range(0, diff_shape[0], batch_size):
            start_time = time.time()  # Start timer for batch processing
            for j in range(0, diff_shape[1], batch_size):
               
                # Define the slice for this batch
                batch_slice = (
                    slice(i, min(i + batch_size, diff_shape[0])),
                    slice(j, min(j + batch_size, diff_shape[1]))
                )

                print(f"Processing batch at position ({i}, {j})...")
                
                # Get the batch of differences
                diff_batch = image2[batch_slice] - image1[batch_slice]
                abs_diff = np.abs(diff_batch)
                
                # Absolute threshold
                abs_threshold = np.std(diff_batch[~np.isnan(diff_batch)])
                abs_changes[batch_slice] = np.where(diff_batch < -abs_threshold, -1, abs_changes[batch_slice])
                abs_changes[batch_slice] = np.where(diff_batch > abs_threshold, 1, abs_changes[batch_slice])

                # Statistical method (z-score per batch)
                valid_data = diff_batch[~np.isnan(diff_batch)]
                if len(valid_data) > 0:
                    z_scores = stats.zscore(valid_data)
                    z_scores_resized = np.zeros_like(diff_batch)
                    z_scores_resized[~np.isnan(diff_batch)] = z_scores
                    stat_changes[batch_slice] = np.where(z_scores_resized < -2, -1, stat_changes[batch_slice])
                    stat_changes[batch_slice] = np.where(z_scores_resized > 2, 1, stat_changes[batch_slice])

                # Percentile method (per batch)
                if len(valid_data) > 0:
                    lower_percentile, upper_percentile = np.percentile(valid_data, [10, 90])
                    perc_changes[batch_slice] = np.where(diff_batch < lower_percentile, -1, perc_changes[batch_slice])
                    perc_changes[batch_slice] = np.where(diff_batch > upper_percentile, 1, perc_changes[batch_slice])
                
                # Otsu method (per batch)
                valid_abs_diff = abs_diff[~np.isnan(abs_diff)]
                if len(valid_abs_diff) > 0:
                    otsu_threshold = threshold_otsu(valid_abs_diff)
                    otsu_changes[batch_slice] = np.where(diff_batch < -otsu_threshold, -1, otsu_changes[batch_slice])
                    otsu_changes[batch_slice] = np.where(diff_batch > otsu_threshold, 1, otsu_changes[batch_slice])

                # Explicit garbage collection after each batch
                del diff_batch, abs_diff, valid_data, z_scores_resized
                # gc.collect()

            end_time = time.time()  # End timer for batch processing
            print(f"Processed batch at position ({i}, {j}) in {end_time - start_time:.2f} seconds.")

        results['Absolute'] = abs_changes
        results['Statistical'] = stat_changes
        results['Percentile'] = perc_changes
        results['Otsu'] = otsu_changes
        
        print("Finished processing all batches.")
        return results

    except Exception as e:
        logging.error(f"Error in change detection: {str(e)}")
        return None



# @st.cache_data
def analyze_results(changes_dict,outputpath):
    """
    Analyze results from all methods.
    Cached to improve performance.
    """
    if changes_dict is None:
        return None
        
    results = []
    for method, changes in changes_dict.items():
        try:
            total_pixels = np.sum(~np.isnan(changes))
            loss_pixels = np.sum(changes == -1)
            gain_pixels = np.sum(changes == 1)
            no_change_pixels = np.sum(changes == 0)
            
            results.append({
                'Method': method,
                'Forest Loss (%)': (loss_pixels / total_pixels) * 100,
                'Forest Gain (%)': (gain_pixels / total_pixels) * 100,
                'No Change (%)': (no_change_pixels / total_pixels) * 100,
                'Loss/Gain Ratio': loss_pixels / max(gain_pixels, 1),
                'Total Changed (%)': ((loss_pixels + gain_pixels) / total_pixels) * 100
            })
        except Exception as e:
            st.warning(f"Error analyzing {method}: {str(e)}")
            continue
    df = pd.DataFrame(results) 
    df.to_csv(f'{outputpath}/change statistical metrics.csv')
    return df


async def save_comparison_to_tiff(changes_dict, output_dir, reference_profile, chunk_size=(32, 32), compress=True, rgb=True):
    """
    Save comparison results as RGB GeoTIFF files with pure RGB colors for changes.
    
    Args:
        changes_dict (dict): Dictionary with change detection results for each method
        output_dir (str): Directory where the images will be saved
        reference_profile (dict): Rasterio profile containing geospatial metadata
        chunk_size (tuple): The size of the chunks to split the image into
        compress (bool): Whether to apply compression to the files
        rgb (bool): Whether to save as RGB (3-band) or grayscale (1-band)
    """
    if changes_dict is None or reference_profile is None:
        print("Error: Missing input data or profile information.")
        return None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    saved_files = []

    def create_rgb_from_changes(change_data):
        """
        Convert change values to pure RGB colors
        Red: Loss (-1) -> (1, 0, 0)
        Green: Gain (1) -> (0, 1, 0)
        White: No change (0) -> (1, 1, 1)
        """
        # Initialize with zeros to ensure no background colors
        rgb = np.zeros((3, change_data.shape[0], change_data.shape[1]), dtype=np.float32)
        
        # Create binary masks for each category
        loss_mask = (change_data == -1)
        gain_mask = (change_data == 1)
        no_change_mask = (change_data == 0)
        
        # Pure red for loss (RGB: 1, 0, 0)
        rgb[0, :, :][loss_mask] = 1.0  # Red channel only
        
        # Pure green for gain (RGB: 0, 1, 0)
        rgb[1, :, :][gain_mask] = 1.0  # Green channel only
        
        # Pure white for no change (RGB: 1, 1, 1)
        rgb[0, :, :][no_change_mask] = 1.0  # Red channel
        rgb[1, :, :][no_change_mask] = 1.0  # Green channel
        rgb[2, :, :][no_change_mask] = 1.0  # Blue channel
        
        # Ensure no other values exist
        rgb = np.clip(rgb, 0, 1)
        
        return rgb

    try:
        for method, changes in changes_dict.items():
            output_file = os.path.join(output_dir, f"{method}_changes.tif")
            height, width = changes.shape
            print(f"Processing {method} results: {height}x{width}")
            
            # Prepare output profile
            output_profile = reference_profile.copy()
            output_profile.update({
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 3 if rgb else 1,
                'dtype': 'float32',
                'compress': 'lzw' if compress else None,
                'tiled': True,
                'blockxsize': chunk_size[1],
                'blockysize': chunk_size[0],
                'BIGTIFF': 'YES',
                'nodata': None,  # Explicitly set to None to prevent interference
                'photometric': 'RGB'  # Ensure correct color interpretation
            })
            
            n_chunks_h = (height + chunk_size[0] - 1) // chunk_size[0]
            n_chunks_w = (width + chunk_size[1] - 1) // chunk_size[1]
            total_chunks = n_chunks_h * n_chunks_w
            chunks_processed = 0
            
            with rasterio.open(output_file, 'w', **output_profile) as dst:
                for i in range(0, height, chunk_size[0]):
                    chunk_height = min(chunk_size[0], height - i)
                    
                    for j in range(0, width, chunk_size[1]):
                        chunk_width = min(chunk_size[1], width - j)
                        
                        try:
                            chunk_slice = (slice(i, i + chunk_height),
                                         slice(j, j + chunk_width))
                            chunk_data = changes[chunk_slice].astype(np.float32)
                            
                            if rgb:
                                # Convert to pure RGB colors
                                chunk_rgb = create_rgb_from_changes(chunk_data)
                                dst.write(chunk_rgb,
                                        window=rasterio.windows.Window(j, i, chunk_width, chunk_height))
                            else:
                                dst.write(chunk_data[np.newaxis, :, :],
                                        window=rasterio.windows.Window(j, i, chunk_width, chunk_height))
                            
                            chunks_processed += 1
                            if chunks_processed % 100 == 0:
                                print(f"Processed {chunks_processed}/{total_chunks} chunks ({(chunks_processed/total_chunks)*100:.1f}%)")
                            
                        except Exception as chunk_error:
                            print(f"Error processing chunk at position ({i}, {j}): {str(chunk_error)}")
                            continue
                        finally:
                            del chunk_data
                            gc.collect()
            
            saved_files.append(output_file)
            print(f"Saved {method} to {output_file}")
            
            # Create a colormap reference file
            colormap_file = os.path.join(output_dir, f"{method}_colormap.txt")
            with open(colormap_file, 'w') as f:
                f.write("Change Detection Color Reference:\n")
                f.write("Pure Red (RGB: 1,0,0): Forest Loss\n")
                f.write("Pure Green (RGB: 0,1,0): Forest Gain\n")
                f.write("Pure White (RGB: 1,1,1): No Change\n")
            
            gc.collect()
        
        print(f"Successfully saved all files to {output_dir}")
        return saved_files
        
    except Exception as e:
        print(f"Error saving files: {str(e)}")
        for file in saved_files:
            try:
                os.remove(file)
            except:
                pass
        return None

        
def plot_comparison(changes_dict, img1, img2):
    """
    Create comparison plots for all methods.
    """
    if any(x is None for x in [changes_dict, img1, img2]):
        return None
        
    try:
        methods = list(changes_dict.keys())
        n_methods = len(methods)
        
        fig = plt.figure(figsize=(15, 4 * (n_methods + 1)))
        gs = plt.GridSpec(n_methods + 1, 3)
        
        # Plot original images
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img1, cmap='gray')
        ax1.set_title('First Image')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img2, cmap='gray')
        ax2.set_title('Second Image')
        
        # Plot difference map
        ax3 = fig.add_subplot(gs[0, 2])
        diff_img = img2 - img1
        ax3.imshow(diff_img, cmap='RdBu')
        ax3.set_title('Difference Map')
        
        # Plot results from each method
        for i, (method, changes) in enumerate(changes_dict.items(), 1):
            ax = fig.add_subplot(gs[i, :])
            im = ax.imshow(changes, cmap='RdYlBu', vmin=-1, vmax=1)
            ax.set_title(f'{method} Method')
            plt.colorbar(im, ax=ax, label='Change Class')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error in plotting: {str(e)}")
        return None

# def main():
#     st.set_page_config(page_title="Forest Change Detection", layout="wide")
    
#     st.title('Forest Change Detection Analysis')
#     st.write("Upload two GeoTIFF images to analyze forest changes between them.")
    
#     # Add file size limit warning
#     st.info("Note: Maximum file size is 200MB per image. Larger files may cause performance issues.")
    
#     # File upload columns
#     col1, col2 = st.columns(2)
#     with col1:
#         image1 = st.file_uploader("First temporal image (.tif)", type=['tif'])
#     with col2:
#         image2 = st.file_uploader("Second temporal image (.tif)", type=['tif'])
    
#     if image1 is not None and image2 is not None:
#         with st.spinner("Processing images..."):
#             try:
#                 # Process images using safe file handling
#                 with safe_file_handle(image1) as file1, safe_file_handle(image2) as file2:
#                     # Preprocess images
#                     img1_processed, profile1 = preprocess_image(file1)
#                     img2_processed, profile2 = preprocess_image(file2)
                    
#                     if img1_processed is None or img2_processed is None:
#                         st.error("Error processing images. Please check the file format and try again.")
#                         return
                    
#                     # Check image dimensions
#                     if img1_processed.shape != img2_processed.shape:
#                         st.error("Images must have the same dimensions.")
#                         return
                    
#                     # Detect changes
#                     changes_dict = detect_changes_multi_method(img1_processed, img2_processed)
                    
#                     if changes_dict is None:
#                         st.error("Error detecting changes. Please try again.")
#                         return
                    
#                     # Analyze and display results
#                     results_df = analyze_results(changes_dict)
                    
#                     if results_df is not None:
#                         st.write("### Change Detection Results")
#                         st.dataframe(results_df)
                        
#                         # Create tabs for different visualizations
#                         tab1, tab2 = st.tabs(["Change Maps", "Method Agreement"])
                        
#                         with tab1:
#                             fig_maps = plot_comparison(changes_dict, img1_processed, img2_processed)
#                             if fig_maps is not None:
#                                 st.pyplot(fig_maps)
                        
#                         with tab2:
#                             # Method agreement analysis
#                             methods = list(changes_dict.keys())
#                             agreement_matrix = np.zeros((len(methods), len(methods)))
                            
#                             for i, method1 in enumerate(methods):
#                                 for j, method2 in enumerate(methods):
#                                     agreement = np.mean(changes_dict[method1] == changes_dict[method2]) * 100
#                                     agreement_matrix[i, j] = agreement
                            
#                             fig_agreement = plt.figure(figsize=(10, 8))
#                             sns.heatmap(agreement_matrix,
#                                       xticklabels=methods,
#                                       yticklabels=methods,
#                                       annot=True,
#                                       fmt='.1f',
#                                       cmap='YlGnBu')
#                             plt.title('Method Agreement Matrix (%)')
#                             st.pyplot(fig_agreement)
                
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")
#                 st.write("Please try again with different images or contact support if the problem persists.")
current_dir = os.getcwd()

# Append "output" to the current directory


async def main():
    image1_path = r"/media/martin/LINUX MINT/Sub_comp50a_ECW/25-06-2024.tif"
    image2_path = r"/media/martin/LINUX MINT/Sub_comp50a_ECW/26-06-2024.tif"
    outputpath = f"{image1_path.split('/')[-1].split('.')[0]} and {image2_path.split('/')[-1].split('.')[0]}"

    if image1_path is not None and image2_path is not None:
        try:
            # Process images
            img1_processed, profile1 = await process_in_batches(image1_path)
            if img1_processed is None:
                print("Error loading first image")
                exit()

            reference_shape = img1_processed.shape
            print("Reference Shape:", reference_shape)

            # Process second image with the same shape
            img2_processed, profile2 = await process_in_batches(image2_path, reference_shape=reference_shape)

            if img2_processed is None:
                print("Error loading second image")
                exit()

            print("Processed Image 1 Shape:", img1_processed.shape)
            print("Processed Image 2 Shape:", img2_processed.shape)
            
            # Detect changes
            changes_dict = await detect_changes_multi_method(img1_processed, img2_processed)
            
            if changes_dict is None:
                print("Error detecting changes. Please try again.")
                return
            print("changed",changes_dict)

            output_dir = os.path.join(current_dir, outputpath)
            os.makedirs(output_dir)
            
            # Analyze and display results
            results_df = analyze_results(changes_dict,output_dir)
            print("hello",results_df)
            
            if results_df is not None:
                print("### Change Detection Results")
                # st.dataframe(results_df)
                
            #     # Create tabs for different visualizations
            #     # tab1, tab2 = st.tabs(["Change Maps", "Method Agreement"])
            await save_comparison_to_tiff(
                changes_dict,
                output_dir=output_dir,
                reference_profile=profile1,  # Using profile from the first image
                chunk_size=(256, 256),
                compress=False
            )
                
            #     # with tab1:
                # fig_maps = plot_comparison(changes_dict, img1_processed, img2_processed)
                # if fig_maps is not None:
                #     plt.pyplot(fig_maps)
                
            #     # with tab2:
            #         # Method agreement analysis
            #     methods = list(changes_dict.keys())
            #     agreement_matrix = np.zeros((len(methods), len(methods)))
                
            #     for i, method1 in enumerate(methods):
            #         for j, method2 in enumerate(methods):
            #             agreement = np.mean(changes_dict[method1] == changes_dict[method2]) * 100
            #             agreement_matrix[i, j] = agreement
                
            #     fig_agreement = plt.figure(figsize=(10, 8))
            #     sns.heatmap(agreement_matrix,
            #                 xticklabels=methods,
            #                 yticklabels=methods,
            #                 annot=True,
            #                 fmt='.1f',
            #                 cmap='YlGnBu')
            #     plt.title('Method Agreement Matrix (%)')
            #     plt.pyplot(fig_agreement)
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again with different images or contact support if the problem persists.")

if __name__ == "__main__":
    asyncio.run(main())
