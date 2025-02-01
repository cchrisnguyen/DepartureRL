import numpy as np
from datetime import datetime, timedelta
import csv
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

from .aircraft import Aircraft
from .utils.enums import Config

def great_circle_distance(lat1, long1, lat2, long2):
    """
    Calculate the great circle distance between two points on the Earth's surface.
    
    Parameters:
    - lat1 (float): Latitude of the first point in degrees
    - long1 (float): Longitude of the first point in degrees
    - lat2 (float): Latitude of the second point in degrees
    - long2 (float): Longitude of the second point in degrees
    
    Returns:
    - float: The great circle distance between the two points in kilometers
    """
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(np.square(np.sin((np.deg2rad(lat2) - np.deg2rad(lat1))/2.0)) + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.square(np.sin((np.deg2rad(long2) - np.deg2rad(long1))/2.0))))

class Departure:
    state_dim = 7
    action_dim = 3

    def __init__(self, task_name="departure", time=datetime(2020, 2, 2)):
        """
        Initialize the Departure environment.
        
        Parameters:
        - task_name (str): The name of the task
        - time (datetime): The initial time for simulation purposes only
        """
        self.task_name = task_name
        self.time=time
        # Initial conditions
        self.initial_lat, self.initial_lon, self.initial_alt, self.initial_cas, self.initial_heading = 22.343216667,  114.027533333, 4500, 205, 80 
        self.initial_path_angle, self.initial_bank_angle = 1, 0
        self.initial_vertical_rate = 10
        self.aircraft = Aircraft(call_sign="RM6111", aircraft_type="B789",
                                lat=self.initial_lat, lon=self.initial_lon, alt=self.initial_alt, 
                                heading=self.initial_heading, path_angle=self.initial_path_angle,
                                bank_angle=self.initial_bank_angle, cas=self.initial_cas,
                                vs=self.initial_vertical_rate, config=Config.CLEAN,
                                fuel_weight=20000.0, payload_weight=50000.0)
        
        # Data
        self.map_corners = {'left':113.834582893, 'bottom':22.152916779, 'right':114.441249557, 'top':22.562083444}
        
        elevation = loadmat(Path(__file__).parent.resolve().joinpath("./data/elevation.mat"))
        self.elevation = elevation["elevation"]
        
        self.population = np.load(Path(__file__).parent.resolve().joinpath("./data/population_density.npy"))
        
        nfz_penalty = loadmat(Path(__file__).parent.resolve().joinpath("./data/NFZ.mat"))
        xs = np.linspace(self.map_corners['left'], self.map_corners['right'], 1000)
        ys = np.linspace(self.map_corners['bottom'], self.map_corners['top'], 1000)
        nfz_penalty = nfz_penalty["Z"]

        self.nfz_penalty = RegularGridInterpolator((ys, xs), nfz_penalty, method='nearest')

        # Final conditions
        self.target_lat, self.target_lon, self.target_alt, self.target_cas = 21.97805556, 114.9108333, 20000, 350

        # Noise impact factor
        self.noise_factor = 1.0
        
        alt = np.array([200, 400, 630, 1000, 2000, 4000, 6300, 10000, 16000, 25000, 1e5, 1e6]) * 0.3048
        thrust = np.array([0, 35684, 48263, 60841, 73419, 85997, 98522, 110000]) * 4.448222
        noise = np.array([[0.000000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.0, 0.0],
                  [100.8000, 93.4000, 88.7000, 83.8000, 75.5000, 66.3000, 59.7000, 52.2000, 44.7000, 37.5000, 0.0, 0.0],
                  [102.0000, 94.9000, 90.1000, 85.0000, 76.7000, 67.7000, 61.3000, 54.3000, 47.2000, 40.5000, 0.0, 0.0],
                  [103.8000, 96.9000, 92.1000, 87.0000, 78.7000, 69.9000, 63.6000, 56.9000, 50.0000, 43.3000, 0.0, 0.0],
                  [105.9000, 99.1000, 94.3000, 89.2000, 81.1000, 72.3000, 66.1000, 59.5000, 52.8000, 46.4000, 0.0, 0.0],
                  [108.6000, 101.8000, 97.1000, 92.1000, 84.1000, 75.4000, 69.3000, 62.8000, 56.1000, 50.0000, 0.0, 0.0],
                  [112.5000, 105.8000, 101.2000, 96.3000, 88.4000, 79.6000, 73.4000, 66.3000, 59.3000, 52.1000, 0.0, 0.0],
                  [112.5000, 105.8000, 101.2000, 96.3000, 88.4000, 79.6000, 73.4000, 66.3000, 59.3000, 52.1000, 0.0, 0.0]])
        
        self.noise_model = RegularGridInterpolator((thrust, alt), noise)


        # Time limit 30m
        self.time_limit = 60*30
        self.sd = 0 
        
        # Last records
        self.last_alt, self.last_cas = self.initial_alt, self.initial_cas
        self.last_lat, self.last_lon = self.initial_lat, self.initial_lon
        
    def seed(self, seed):
        """
        Set the seed for the environment's random number generator. Currently not in use.
        
        Parameters:
        - seed (int, optional): The seed value for the random number generator. If None, a random seed will be used.
        """
        self.sd = seed

    def in_circle(self, lat, lon):
        """
        Check if a point is within a circle defined by a center point and radius.
        
        Parameters:
        - lat (float): Latitude of the point to check
        - lon (float): Longitude of the point to check
        
        Returns:
        - bool: True if the point is within the circle, False otherwise
        """
        radius_square = (self.initial_lat - self.target_lat)**2 + (self.initial_lon - self.target_lon)**2
        return (self.initial_lat - lat)**2 + (self.initial_lon - lon)**2 < radius_square
    
    def state_adapter(self, lat, lon, alt, heading, bank, cas):
        """
        Adapt the state to the input range of the model.
        
        Parameters:
        - state (list): The current state of the environment
        
        Returns:
        - np.array: The adapted state
        """
        lat = lat - self.initial_lat
        lon = lon - self.initial_lon
        alt = (alt - 10000)/10000
        cos = np.cos(heading*np.pi/180)
        sin = np.sin(heading*np.pi/180)
        cas = (cas-200)/150
        bank /= 30
        return np.array([lat, lon, alt, cos, sin, bank, cas])
    
    def reset(self, time=datetime(2020, 2, 2), write_csv=False):
        """
        Reset the environment to its initial state.
        
        Parameters:
        - time (datetime): The initial time for simulation purposes only
        - write_csv (bool): Whether to write the environment state to a CSV file
        
        Returns:
        - np.array: The adapted initial state of the environment
        """
        self.global_time = 0
        self.time=time

        self.writer = None
        if write_csv:
            file_name = self.task_name +'.csv'
            file_path = Path(__file__).parent.resolve().joinpath('post_processing/'+file_name)
            self.writer = csv.writer(open(file_path, 'w+'))
            header = ['timestep', 'timestamp', 'id', 'callsign', 
                      'lat', 'long', 'alt', 'cas', 'heading', 'bank_angle', 'path_angle',
                      'mass', 'thrust', 'consumed_fuel', 'noise', 'nfz_penalty']
            self.writer.writerow(header)

        self.aircraft.reset(lat=self.initial_lat, lon=self.initial_lon, alt=self.initial_alt, 
                                        heading=self.initial_heading, path_angle=self.initial_path_angle,
                                        bank_angle=self.initial_bank_angle, cas=self.initial_cas,
                                        vs=self.initial_vertical_rate, config=Config.CLEAN,
                                        fuel_weight=20000.0, payload_weight=50000.0)
        
        # Last records
        self.last_alt, self.last_cas = self.initial_alt, self.initial_cas
        self.last_lat, self.last_lon = self.initial_lat, self.initial_lon

        return self.state_adapter(self.initial_lat, self.initial_lon, self.initial_alt,
                                   self.initial_heading, self.initial_bank_angle, self.initial_cas)

    def step(self, action):
        assert len(action) == 3

        total_fuel = 0
        total_noise = 0
        total_penalty = 0
        dead = False
        for _ in range(10):
            self.aircraft.change_heading(2*action[0])       # min: -3deg/s, max: 3deg/s
            self.aircraft.change_cas(2*(action[1]+1))       # max: 4 knots/s
            self.aircraft.change_altitude(20*(action[2]+1)) # max: 40 ft/s
            lat, lon, alt, head, bank, cas, fuel, thrust =  self.aircraft.update()
            self.global_time += 1
            
            total_fuel += fuel
            noise_impact = self.noise_pollution_level(lat, lon, alt, thrust)
            total_noise += noise_impact
            nfz_penalty = self.compute_nfz_penalty(lat, lon)
            total_penalty += nfz_penalty

            if not self.in_circle(lat, lon):
                dead = True
                break
                
            self.save(lat, lon, alt, head, bank, cas, thrust, noise_impact, nfz_penalty)

        if dead:
            distance_gain = great_circle_distance(lat, lon, self.last_lat, self.last_lon)
        else:
            distance_gain = great_circle_distance(self.last_lat, self.last_lon, self.target_lat, self.target_lon) \
                            - great_circle_distance(lat, lon, self.target_lat, self.target_lon)

        reward = -total_fuel/10 - self.noise_factor*total_noise - total_penalty + \
                    2*distance_gain + (alt - self.last_alt)/80 + (cas - self.last_cas)/4

        # Update
        self.last_alt, self.last_cas = alt, cas
        self.last_lat, self.last_lon = lat, lon
        
        # Check time out
        terminated = False
        if (self.global_time >= self.time_limit) or dead:
            terminated = True
        
        return self.state_adapter(lat, lon, alt, head, bank, cas), reward, terminated, dead, (self.global_time, total_fuel, noise_impact, total_penalty)

    def step_evaluate(self, action):
        assert len(action) == 3
        dead = False
        total_fuel = 0
        total_noise = 0
        total_penalty = 0
        for _ in range(10):
            self.aircraft.change_heading(2*action[0])       # min: -3deg/s, max: 3deg/s
            self.aircraft.change_cas(2*(action[1]+1))       # max: 4 knots/s
            self.aircraft.change_altitude(20*(action[2]+1)) # max: 40 ft/s
            lat, lon, alt, head, bank, cas, fuel, thrust =  self.aircraft.update()
            self.global_time += 1
            noise_impact = self.noise_pollution_level(lat, lon, alt, thrust)
            nfz_penalty = self.compute_nfz_penalty(lat, lon)
            print(lat, lon, alt, thrust, noise_impact, nfz_penalty)
            self.save(lat, lon, alt, head, bank, cas, thrust, noise_impact, nfz_penalty)
            total_fuel += fuel
            total_noise += noise_impact
            total_penalty += nfz_penalty
            if not self.in_circle(lat, lon):
                dead = True
                break
            
        # Check out of time
        terminated = False
        if (self.global_time >= self.time_limit) or dead:
            terminated = True
        
        return self.state_adapter(lat, lon, alt, head, bank, cas), terminated, (total_fuel, total_noise, total_penalty)

    def save(self, lat, lon, alt, head, bank, cas, thrust, noise, nfz_penalty):
        """
        Save all states variable of one timestemp to csv file.
        """
        if self.writer:
            data = [self.global_time, (self.time + timedelta(seconds=self.global_time)).isoformat(timespec='seconds'), 0, self.aircraft.call_sign, 
                    lat, lon, alt, cas, head, bank, self.aircraft.path_angle,
                    self.aircraft.mass, thrust, self.aircraft.consumed_fuel, noise, nfz_penalty]
            
            self.writer.writerow(data)
    
    def noise_pollution_level(self, latitude, longitude, altitude, thrust, threshold=50):
        altitude *= 0.3048  # convert ft to m, from now on, all SI unit
        match threshold:
            case 50:
                impact_radius = 8602.26  # 60 dB ==> 4708.6 m
            case 55:
                impact_radius = 6402.21  # 60 dB ==> 4708.6 m
            case 60:
                impact_radius = 4.7086e3  # 60 dB ==> 4708.6 m
            case 65:
                impact_radius = 3.3594e3  # 65 dB ==> 3359.4 m
            case 70:
                impact_radius = 2.3686e3  # 70 dB ==> 2368.6 m
            case 75:
                impact_radius = 1.6490e3  # 75 dB ==> 1649.0 m
            case _:
                ValueError("Currently arbitrary threshold is not supported.")

        if altitude >= impact_radius:
            return 0.0

        search_radius = (impact_radius ** 2 - altitude ** 2) ** 0.5

        nx = 728  # number of cells in x (EW) direction
        ny = 491  # number of cells in y (NS) direction
        dx = 85.714  # 62400 / 728
        dy = 92.281  # 45310 / 491
        ix = (longitude - self.map_corners['left']) / (self.map_corners['right'] - self.map_corners['left']) * nx  # position in i (has decimal)
        iy = (self.map_corners['top'] - latitude) / (self.map_corners['top'] - self.map_corners['bottom']) * ny  # position in j (has decimal)
        sx = search_radius / dx  # number of cells within radius in x direction
        sy = search_radius / dy  # number of cells within radius in y direction

        lb = min(max(round(ix - sx), 0), nx-1)  # left bound of impact region
        rb = min(max(round(ix + sx), 0), nx-1)  # right bound of impact region
        tb = min(max(round(iy - sy), 0), ny-1)  # top bound of impact region
        bb = min(max(round(iy + sy), 0), ny-1)  # bottom bound of impact region
        ixs, iys = np.meshgrid(np.arange(lb, rb + 1), np.arange(tb, bb + 1))

        impact_rectangle = self.population[tb:bb + 1, lb:rb + 1].flatten()
        impact_distance = np.sqrt(
            np.square(np.abs(ixs - ix) * dx) +
            np.square(np.abs(iys - iy) * dy) +
            np.square(altitude - self.elevation[tb:bb + 1, lb:rb + 1]) + 1
        ).flatten()
        impact_noise = self.noise_model(np.vstack([np.full(len(impact_distance), thrust/2), impact_distance]).T)
        impact_noise = np.maximum(impact_noise - threshold, 0)

        influence = np.dot(impact_noise, impact_rectangle)/1e6
        return influence
    
    def compute_nfz_penalty(self, lat, lon):
        if (lat < self.map_corners['bottom']) or (lat > self.map_corners['top']) or (lon < self.map_corners['left']) or (lon > self.map_corners['right']):
            return 0
        return self.nfz_penalty([lat, lon])[0]
    