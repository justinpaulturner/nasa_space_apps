import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import geopandas as gpd
from geopandas import GeoDataFrame
from datetime import datetime


class Fire:
    def __init__(self, danger_radius=20, sample_size=1000):
        self.school_df = pd.read_csv("data/school data/Public_Schools.csv")

        if sample_size:
            self.school_df = self.school_df.sample(sample_size)

        self.school_df = self.school_df[(self.school_df['COUNTRY']=='USA')&(self.school_df['STATE']!='AK')&(self.school_df['STATE']!='HI')]
        # self.school_df = pd.DataFrame(self.school_df[['NAME','LATITUDE','LONGITUDE']])
        self.school_df.rename(columns=str.lower, inplace=True)
        self.fire_df = pd.read_csv('https://firms.modaps.eosdis.nasa.gov/data/active_fire/c6/csv/MODIS_C6_USA_contiguous_and_Hawaii_24h.csv')

        # self.fire_df = self.fire_df.iloc[150:250, :]

        self.danger_radius = danger_radius
        self.all_distances = None
        
    def current_hour(self):
        """ 
        Returns the integer of the number of the current hour
        """
        return int(datetime.now().strftime("%H"))

    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles
        return c * r


    def proximity(self, points):
        lat1 = points[0]
        lon1 = points[1]
        lat2 = points[2]
        lon2 = points[3]

        return self.haversine(lon1, lat1, lon2, lat2)

    def all_flat_combinations(self, a1, a2):
        m1,n1 = a1.shape
        m2,n2 = a2.shape
        out = np.zeros((m1,m2,n1+n2),dtype=float)
        out[:,:,:n1] = a1[:,None,:]
        out[:,:,n1:] = a2
        out.shape = (m1*m2,-1)
        return out
    
    def schools_in_danger_zone(self):
        school_arr = self.school_df[['latitude', 'longitude']].values
        fire_arr = self.fire_df[['latitude', 'longitude']].values

        afc = self.all_flat_combinations(school_arr, fire_arr)
        print('all combinations generated')
        self.all_distances = np.apply_along_axis(self.proximity, 1, afc)
        print('all distances calculated')

        afc_len = afc.shape[0]
        n_fires = fire_arr.shape[0]
        
        return [j for j, i in enumerate(range(0, afc_len, n_fires))
                if np.any(self.all_distances[i: i+n_fires] < self.danger_radius)]

    
    def generate_schools_in_danger(self):
        """Finds locations of schools in danger, and indicies 'school_df' to create 'self.danger_df'."""
        danger_indices = self.schools_in_danger_zone()
        self.danger_df = self.school_df.iloc[danger_indices]
        
    def save_schools(self):
        """overwrites the school_list.txt file with the updated list of schools in danger."""
        # code to open the school_list.txt file and overwrite the schools with the new schools in danger
        self.danger_df.to_csv('flaskapp/flaskr/static/school_list.txt', sep=' ', index=False)
        
    def generate_new_image(self):
        """Generates the new image of just the schools within 2 km of a fire."""
        geometry = [Point(xy) for xy in zip(self.danger_df['longitude'], self.danger_df['latitude'])]
        gdf = GeoDataFrame(self.danger_df, geometry=geometry)   

        #this is a simple map that goes with geopandas
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world_plot = world.plot(figsize=(10, 6))
        world_plot.set_xlim([-125,-65])
        world_plot.set_ylim([25,50])
        world_plot.set_ylabel("Latitude")
        world_plot.set_xlabel("Longitude")
        fig = gdf.plot(ax=world_plot, marker='o', color='red', markersize=10).get_figure()
        fig.savefig("flaskapp/flaskr/static/image.png")
        
        