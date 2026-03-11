# This code extracts streamflows from netcdf files and saves them as csv files for each feature ID (gauge) separately. Date is sorted.

import xarray as xr
import pandas as pd
import os
import numpy as np

def extract_streamflow_by_feature_chunks(nc_file_path, output_dir, chunk_size=100):
    """
    Extract streamflow data from NetCDF file in chunks, 
    average by day for each feature ID, and save as separate CSV files.
    
    Parameters:
    -----------
    nc_file_path : str
        Path to the input NetCDF file
    output_dir : str
        Directory to save output CSV files
    chunk_size : int, optional
        Number of feature IDs to process in each chunk
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the NetCDF file
    ds = xr.open_dataset(nc_file_path)
    
    # Get unique feature IDs
    all_feature_ids = ds['feature_id'].values
    
    # Process feature IDs in chunks
    for i in range(0, len(all_feature_ids), chunk_size):
        # Select a chunk of feature IDs
        chunk_feature_ids = all_feature_ids[i:i+chunk_size]
        
        # Select data for this chunk of feature IDs
        chunk_ds = ds.sel(feature_id=chunk_feature_ids)
        
        # Convert to DataFrame
        chunk_df = chunk_ds.to_dataframe().reset_index()
        
        # Convert time to datetime
        chunk_df['time'] = pd.to_datetime(chunk_df['time'])
        
        # Group by feature_id and date, then average streamflow
        daily_avg = chunk_df.groupby([chunk_df['feature_id'], chunk_df['time'].dt.date])['streamflow'].mean().reset_index()
        
        # Rename date column for clarity
        daily_avg = daily_avg.rename(columns={'time': 'date'})
        
        # Save CSV for each feature ID in this chunk
        for feature_id in chunk_feature_ids:
            # Filter data for specific feature ID
            feature_data = daily_avg[daily_avg['feature_id'] == feature_id].sort_values('date')
            
            # Create output filename
            output_filename = os.path.join(output_dir, f'{feature_id}.csv')
            
            # Save to CSV
            feature_data.to_csv(output_filename, index=False)
            print(f"Saved data for feature ID {feature_id} to {output_filename}")
        
        # Clear memory
        del chunk_ds, chunk_df, daily_avg
    
    # Close the dataset
    ds.close()

# Specify the input file path and output directory
nc_file_path = '/netfiles/ciroh/rvanderh/NWM Downloading/NWM Download Scripts/NWM_V3.0_GW_Outflow.nc'

output_directory = '/netfiles/ciroh/rvanderh/NWM Downloading/NWM Download Scripts/NWM_v3.0_GW_Outflow_CSVs/'

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Run the extraction with chunked processing
extract_streamflow_by_feature_chunks(nc_file_path, output_directory, chunk_size=100)

# Optional: Print total number of files created
print(f"Total CSV files created in {output_directory}")
print(f"Number of files: {len(os.listdir(output_directory))}")