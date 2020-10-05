import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 
Imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, Polygon
import geopandas as gpd
from geopandas import GeoDataFrame



class Fire:
    def __init__(self):
        self.school_df = pd.read_csv("data/school data/Public_Schools.csv")
        self.school_df = self.school_df[(self.school_df['COUNTRY']=='USA')&(self.school_df['STATE']!='AK')&(self.school_df['STATE']!='HI')]
        self.school_df = pd.DataFrame(self.school_df[['NAME','LATITUDE','LONGITUDE']])
        
    def distance_to_fire(self,fire_location,school_location):
        # approximate radius of earth in km
        R = 6373.0

        lat1 = radians(fire_location[0])
        lon1 = radians(fire_location[1])
        lat2 = radians(school_location[0])
        lon2 = radians(school_location[1])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c

        return distance
    
    def generate_schools_in_danger(self):
        """Runs Jared's code to save a dataframe that identifies the schools in danger."""
        # Jared's code that uses numpy
        # saves a danger_schools_df or adds an 'in-dnager' column to self.school_df with bool values.
        
    def generate_new_image(self):
        """Generates the new image of just the schools within 2 km of a fire."""
        geometry = [Point(xy) for xy in zip(self.school_df[self.school_df['in-danger']==1]['LONGITUDE'], self.school_df[self.school_df['in-danger']==1]['LATITUDE'])]
        gdf = GeoDataFrame(df, geometry=geometry)   

        #this is a simple map that goes with geopandas
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world_plot = world.plot(figsize=(10, 6))
        world_plot.set_xlim([-125,-65])
        world_plot.set_ylim([25,50])
        world_plot.set_ylabel("Latitude")
        world_plot.set_xlabel("Longitude")
        fig = gdf.plot(ax=world_plot, marker='o', color='red', markersize=1).get_figure()
        fig.savefig("flaskapp/flaskr/static/image.png")