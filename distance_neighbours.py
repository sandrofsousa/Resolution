from csv import reader
from math import sin, cos, sqrt, atan2, radians
from time import time

start = time()


def get_zones_coordinates(path):
    """
    Function to read file as input and get latitude and longitude from zones using a simple parsing,
    output a list with all stops and its respective coordinates. Data must be converted to latitude
    and longitude format WGS84.
    """
    coordinates = []

    with open(path, "r", newline='') as data:
        # parse data using csv based on ',' position.
        searcher = reader(data, delimiter=',', quotechar='"')
        # skip header (first line).
        next(searcher)
        for line in searcher:  # Select the respective column of line based on ',' position
            zone = int(line[0])
            zone_lat = float(line[-2])
            zone_lon = float(line[-1])

            # append result to the list
            coordinates.append((zone, zone_lat, zone_lon))

    return coordinates


def distance_on_sphere(lat1, lon1, lat2, lon2):
    """
    Auxiliary function to calculate distance on sphere in meters from two latitude and longitude pars,
    output the distance in meters of two given coordinates. based on John D. Cock algorithm.
    """
    r = 6371007.176   # Approximate mean radius of earth in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])  # Convert decimal degrees to radians
    # Compute difference from variables
    dif_lat = lat2 - lat1
    dif_lon = lon2 - lon1

    # Haversine formula to calculate the great-circle distance between two points
    a = sin(dif_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dif_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = r * c
    return distance


def get_neighbors(radius, zones_list):
    """
    Function to search zones near each other (neighbors). A radius value and a list of stops with coordinates
    are passed as input. A zone is classified as neighbor if the distance is lower than the radius, outputing
    a dictionary with zone as key and IDs of its neighbors.
    """
    neighbors = dict()
    for row in range(len(zones_list)):  # Populate dictionary with keys and empty values for content
        zone = zones_list[row][0]
        neighbors[zone] = []

    for row1 in range(len(zones_list) - 1):  # Loop reading the list of stops, positioning in the first line of file
        zone1 = zones_list[row1][0]  # Get values from first row.
        lat1 = zones_list[row1][1]
        lon1 = zones_list[row1][2]

        for row2 in range(row1 + 1, len(zones_list)):  # Read value of stops list, getting the position from row1.
            zone2 = zones_list[row2][0]  # Get values from second row.
            lat2 = zones_list[row2][1]
            lon2 = zones_list[row2][2]
            distance = distance_on_sphere(lat1, lon1, lat2, lon2)  # Calculate the distance between stops.
            # If distance <= rho, update dictionary for respective keys (stop2 is neighbor of stop1, reciprocal).
            if distance <= radius:
                neighbors[zone1].append((zone2, distance))
                neighbors[zone2].append((zone1, distance))
            else:
                continue
    return neighbors


# file_path = "data/resolution_oa_2011_ks201ew_WGS84.csv"     # London 50k
# file_path = "data/AP2010_CEM_RMSP_EGP_EDU_WGS84.csv"        # Sao Paulo 600
file_path = "data/SC2010_CEM_RMSP_Income_Race_WGS84.csv"    # Sao Paulo 30k

geodata = get_zones_coordinates(file_path)
radius = 1000
neighbors = get_neighbors(radius, geodata)
print(len(neighbors))


end = time()
elapsed = ((end - start) / 60) / 60
print("Run time: " + str(elapsed))
