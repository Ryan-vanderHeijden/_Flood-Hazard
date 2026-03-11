import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from dask.diagnostics import ProgressBar

# Path to the CSV file containing NWM_ID and USGS_ID
csv_file_path = '/netfiles/ciroh/rvanderh/NWM Downloading/NWM Download Scripts/NWM_USGS_Natural_Flow.csv'

# Path to the output NetCDF file
# output_file = 'D:/OneDrive - University of Vermont/Documents/_UVM/Research/CIROH/NWM/NWM Download Scripts/Natural_Flow/NWM_V3.0_Streamflow_Daily.nc'
output_file = '/netfiles/ciroh/rvanderh/NWM Downloading/NWM Download Scripts/NWM_V3.0_GW_Outflow.nc'

# Read the CSV file and extract the NWM_ID column
df = pd.read_csv(csv_file_path)

# Ensure that 'NWM_ID' exists in the DataFrame
if 'NWM_ID' in df.columns:
    input_ids = sorted(df['NWM_ID'].astype(int).unique().tolist())
else:
    raise ValueError("Column 'NWM_ID' not found in the CSV file.")

# S3 bucket URI and region name
# bucket_uri = 's3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr'
bucket_uri = 's3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/gwout.zarr'
region_name = 'us-east-1'

# Initialize S3 filesystem with anonymous access
s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=region_name))
s3store = s3fs.S3Map(root=bucket_uri, s3=s3, check=False)

# Open the Zarr dataset from S3
ds = xr.open_zarr(s3store)
print(ds)

# Check which input_ids exist in the Zarr dataset
available_ids = [id_ for id_ in input_ids if id_ in ds.feature_id.values]

if not available_ids:
    raise ValueError("None of the input IDs were found in the Zarr dataset.")

print(f"Available input IDs: {available_ids}")

# Select the 'streamflow' variable, filter by available feature IDs
ds_filtered = ds[['streamflow']].sel(feature_id=available_ids)
# ds_filtered = ds[['outflow']].sel(feature_id=available_ids)

print(ds_filtered)

# Define chunk sizes for the output NetCDF file
out_chunks = {'feature_id': 1, 'time': len(ds_filtered.time)}

# Write the dataset to a NetCDF file with specified encoding options
with ProgressBar():
    ds_filtered.to_netcdf(
        output_file,
        encoding={
            'streamflow': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 3,
                'chunksizes': (1, 30)
            }
        }
    )

# with ProgressBar():
#     ds_filtered.to_netcdf(
#         output_file,
#         encoding={
#             'outflow': {
#                 'dtype': 'float32',
#                 'zlib': True,
#                 'complevel': 3,
#                 'chunksizes': (1, 30)
#             }
#         }
#     )