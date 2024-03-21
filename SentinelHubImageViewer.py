import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType
from evalscript import evalscript

# Set up Sentinel Hub configuration
config = SHConfig()
config.sh_client_id = 'e7d3d66f-15de-4612-bdd3-dfbd345db149'
config.sh_client_secret = 'OMBa8BPSh6Y030TEFZeMKZEzJDGHZw0d'

# Initialize geocoder
geolocator = Nominatim(user_agent="forest_fire_detection")

# Prompt the user to enter a place name
place_name = input("Enter a place name: ")

# Get coordinates from place name
location = geolocator.geocode(place_name)
if location:
    lat = location.latitude
    lon = location.longitude
    print(f"Coordinates for {place_name}: Latitude: {lat}, Longitude: {lon}")
else:
    print("Location not found. Please enter a valid place name.")
    exit()

# Define bounding box
bbox = BBox((lon-1, lat-1, lon+1, lat+1), crs=CRS.WGS84)

# Prompt the user to enter the time period
start_date = input("Enter start date (YYYY-MM-DD): ")
end_date = input("Enter end date (YYYY-MM-DD): ")
time_interval = (start_date, end_date)

# Prompt the user to enter the cloud coverage threshold
cloud_coverage_threshold = float(input("Enter maximum cloud coverage threshold (0-1): "))

# Specify the data collection
data_collection = DataCollection.SENTINEL2_L2A

# Define the request
request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=data_collection,
            time_interval=time_interval,
            maxcc=cloud_coverage_threshold
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.TIFF)
    ],
    bbox=bbox,
    size=(750, 750),
    config=config,
    data_folder='./forest_fire_images'
)

# Get the data
image = request.get_data()[0]

# Visualize the image using Matplotlib
plt.imshow(image)
plt.axis('off')  # Turn off axes
plt.show()