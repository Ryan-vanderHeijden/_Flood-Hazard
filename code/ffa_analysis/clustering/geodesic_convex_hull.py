import numpy as np
from scipy.spatial import ConvexHull
from pyproj import Geod

def calculate_area_sq_km(point_list):
    """
    Calculates the area of a point cloud in square kilometers.
    Input: List of [lon, lat] coordinates
    """
    points = np.array(point_list)
    
    # 1. Find the outer boundary points (Convex Hull)
    hull = ConvexHull(points)
    boundary_points = points[hull.vertices]
    
    # 2. Extract lons and lats
    lons = boundary_points[:, 0]
    lats = boundary_points[:, 1]
    
    # 3. Calculate surface area on the WGS84 ellipsoid
    geod = Geod(ellps="WGS84")
    area_meters, _ = geod.polygon_area_perimeter(lons, lats)
    
    # 4. Convert Square Meters to Square Kilometers
    # (Divide by 1,000,000)
    sq_km = abs(area_meters) / 1e6
    
    return np.around(sq_km, 0)

# Example with a rough area in the Midwest
# coords = [[-95.0, 40.0], [-90.0, 40.0], [-90.0, 35.0], [-95.0, 35.0]]
# print(f"Area: {calculate_area_sq_km(coords):,.2f} km²")