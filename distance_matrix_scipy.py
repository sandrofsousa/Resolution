from csv import reader
import numpy as np
from scipy.spatial.distance import cdist
from time import time
from segregationMetrics import Metrics


def get_zones_coordinates(path):
    """
    Function to read file as input and get latitude and longitude from zones using a simple parsing,
    output a list with all stops and its respective coordinates. Data must be converted to tmerc CRS
    format and putted in X and Y format.
    """
    coordinates = {}

    with open(path, "r", newline='') as data:
        # parse data using csv based on ',' position.
        searcher = reader(data, delimiter=',', quotechar='"')
        # skip header (first line).
        next(searcher)
        for line in searcher:  # Select the respective column of line based on ',' position
            zone = int(line[0])
            zone_lat = float(line[1])
            zone_lon = float(line[2])

            # append result to the list
            coordinates[zone] = [zone_lat, zone_lon]

        data.close()
    return coordinates


# file_path = "data/resolution_oa_2011_ks201ew_WGS84.csv"     # London 50k
# file_path = "data/AP2010_CEM_RMSP_EGP_EDU_WGS84.csv"        # Sao Paulo 600
# file_path = "data/SC2010_CEM_RMSP_Income_Race_WGS84.csv"    # Sao Paulo 30k
file_path = "data/tmerc_edu.csv"    # temporary file converted from sirgas to tmerc

coordinates_dict = get_zones_coordinates(file_path)
coordinates_array = np.array([(val[0], val[1]) for key, val in coordinates_dict.items()])

# matrix_np = distance_on_sphere(coordinates_array)
matrix_sp = cdist(coordinates_array, coordinates_array, metric='euclidean')

print("  Dictionary: ", coordinates_dict.get(1))
print("Numpy  array: ", coordinates_array[0])
print("Array  shape: ", coordinates_array.shape)  # (30815, 2)

print("Matrix shape: ", matrix_sp.shape)

sel = matrix_sp[matrix_sp > 3000]

print(len(sel))

# cc = Metrics()
# cc.readAttributesFile("data/tmerc_edu.csv")
# costMatrix = matrix_sp*1000
# cc.locality = cc.cal_localityMatrix(3000)
# print(cc.locality)


# start = time()
# # np.savetxt('data/MATRIX_AP2010_CEM_RMSP_EGP_EDU_WGS84NP.csv', matrix_np, delimiter=',', newline='\n')
# # np.savetxt('data/MATRIX_AP2010_CEM_RMSP_EGP_EDU_WGS84SP.csv', matrix_sp, delimiter=',', newline='\n')
# # np.savetxt('data/MATRIX_SC2010_CEM_RMSP_Income_Race_WGS84.csv', matrix_sp, delimiter=',', newline='\n')
# # np.savetxt('data/MATRIX_SC2010_CEM_RMSP_Income_Race_WGS84.csv', matrix_sp, delimiter=',', newline='\n')
# end = time()
# elapsed = ((end - start) / 60) / 60
# print("Run time: " + str(elapsed))

# print(matrix_np)

# for i in coordinates_dict.items(): print(i)
# print(coordinates_array[:10])

