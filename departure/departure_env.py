import numpy as np
from datetime import datetime, timedelta
import csv
from pathlib import Path

import os, sys
if os.path.abspath(__file__ + "/../../") not in sys.path:
    sys.path.insert(1, os.path.abspath(__file__ + "/../../"))

from core.aircraft import Aircraft
from core.utils.enums import Config

def great_circle_distance(lat1, long1, lat2, long2):
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(np.square(np.sin((np.deg2rad(lat2) - np.deg2rad(lat1))/2.0)) + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.square(np.sin((np.deg2rad(long2) - np.deg2rad(long1))/2.0))))

class Departure:
    state_dim = 7
    action_dim = 3

    def __init__(self, task_name="departure", time=datetime(2020, 2, 2)):
        self.task_name = task_name
        self.time=time
        # Initial conditions
        self.initial_lat, self.initial_lon, self.initial_alt, self.initial_cas, self.initial_heading = 22.343216667,  114.027533333, 4500, 205, 80 
        self.initial_path_angle, self.initial_bank_angle = 1, 0
        self.initial_vertical_rate = 10
        self.aircraft = Aircraft(current_time=time, call_sign="RM6111", aircraft_type="B789",
                                        lat=self.initial_lat, lon=self.initial_lon, alt=self.initial_alt, 
                                        heading=self.initial_heading, path_angle=self.initial_path_angle,
                                        bank_angle=self.initial_bank_angle, cas=self.initial_cas,
                                        vs=self.initial_vertical_rate, config=Config.CLEAN,
                                        fuel_weight=20000.0, payload_weight=50000.0)
        
        # Population dataset
        self.population = np.load("/home/chris/AirTrafficSim/RL/departure/data/population_density.npy")
        self.population_corners = {'left':113.834582893, 'bottom':22.152916779, 'right':114.441249557, 'top':22.562083444}

        
        # Final conditions
        self.target_lat, self.target_lon, self.target_alt, self.target_cas = 21.97805556, 114.9108333, 20000, 350
        
        # Noise impact factor
        self.noise_factor = 1

        # Time limit 30m
        self.time_limit = 60*30
        self.sd = 0
        
        # Last records
        self.last_alt, self.last_cas = self.initial_alt, self.initial_cas
        self.last_distance = great_circle_distance(self.initial_lat, self.initial_lon, self.target_lat, self.target_lon)

    def seed(self, seed):
        self.sd = seed

    def in_circle(self, lat, lon):
        center_lat = (3*self.initial_lat + 0*self.target_lat)/3
        center_lon = (3*self.initial_lon + 0*self.target_lon)/3
        radius_square = (center_lat - self.target_lat)**2 + (center_lon - self.target_lon)**2
        return (center_lat - lat)**2 + (center_lon - lon)**2 < radius_square
    
    def state_adapter(self, lat, lon, alt, heading, bank, cas):
        lat = lat - self.target_lat
        lon = lon - self.target_lon
        alt = (alt - 10000)/10000
        cos = np.cos(heading*np.pi/180)
        sin = np.sin(heading*np.pi/180)
        cas = (cas-200)/150
        bank /= 30
        return np.array([lat, lon, alt, cos, sin, bank, cas])
    
    def reset(self, time=datetime(2020, 2, 2), write_csv=False):
        self.global_time = 0
        self.time=time

        self.writer = None
        if write_csv:
            file_name = self.task_name+'-'+datetime.utcnow().isoformat(timespec='seconds')
            folder_path = Path(__file__).parent.parent.parent.resolve().joinpath('result/'+file_name)
            folder_path.mkdir()
            file_path =  folder_path.joinpath(file_name+'.csv')
            self.writer = csv.writer(open(file_path, 'w+'))
            header = ['timestep', 'timestamp', 'id', 'callsign', 
                      'lat', 'long', 'alt', 'cas', 'heading', 'bank_angle', 'path_angle',
                      'mass', 'fuel_consumed', 'noise']
            self.writer.writerow(header)

        self.aircraft.reset(current_time=time, lat=self.initial_lat, lon=self.initial_lon, alt=self.initial_alt, 
                                        heading=self.initial_heading, path_angle=self.initial_path_angle,
                                        bank_angle=self.initial_bank_angle, cas=self.initial_cas,
                                        vs=self.initial_vertical_rate, config=Config.CLEAN,
                                        fuel_weight=20000.0, payload_weight=50000.0)
        
        # Last records
        self.last_alt, self.last_cas = self.initial_alt, self.initial_cas
        self.last_distance = great_circle_distance(self.initial_lat, self.initial_lon, self.target_lat, self.target_lon)

        return self.state_adapter(self.initial_lat, self.initial_lon, self.initial_alt,
                                   self.initial_heading, self.initial_bank_angle, self.initial_cas)

    def step(self, action):
        assert len(action) == 3

        total_fuel = 0
        for _ in range(10):
            self.aircraft.change_heading(3*action[0])       # min: -3deg/s, max: 3deg/s
            self.aircraft.change_cas(2*(action[1]+1))       # max: 2 knots/s
            self.aircraft.change_altitude(50*(action[2]+1)) # max: 50 ft/s
            lat, lon, alt, head, bank, cas, fuel, _ =  self.aircraft.update()
            self.global_time += 1
            # dump to csv
            self.save(lat, lon, alt, head, bank, cas, fuel)
            total_fuel += fuel
            
            if not self.in_circle(lat, lon):
                distance_to_target = great_circle_distance(lat, lon, self.target_lat, self.target_lon)
                noise_impact = self.noise_pollution_level(lat, lon, alt)
                reward = -total_fuel/10 - self.noise_factor*noise_impact +\
                        3*(self.last_distance - distance_to_target) + 3*(10/(distance_to_target+1) - distance_to_target) +\
                        (alt - self.last_alt)/500 - abs(alt-self.target_alt)/10 +\
                        (cas - self.last_cas)/20 - abs(cas-self.target_cas)

                return self.state_adapter(lat, lon, alt, head, bank, cas), reward, True, True, (self.global_time, total_fuel, noise_impact)
        
        
        distance_to_target = great_circle_distance(lat, lon, self.target_lat, self.target_lon)
        noise_impact = self.noise_pollution_level(lat, lon, alt)
        reward = -total_fuel/10 - self.noise_factor*noise_impact +\
                    3*(self.last_distance - distance_to_target) +\
                    (alt - self.last_alt)/500 +\
                    (cas - self.last_cas)/20

        # print(self.last_cas, cas)
        # print(f"Reward {reward}: fuel {total_fuel}, noise {noise_impact}, distance {self.last_distance - distance_to_target}, alt {(alt - self.last_alt)/3281}, cas {(cas - self.last_cas)/20}")

        # Update
        self.last_alt, self.last_cas = alt, cas
        self.last_distance = distance_to_target
        
        # Check out of time
        terminated = False
        if self.global_time >= self.time_limit:
            terminated = True
        
        return self.state_adapter(lat, lon, alt, head, bank, cas), reward, terminated, False, (self.global_time, total_fuel, noise_impact)
    
        # self.aircraft.change_heading(3*action[0])       # min: -3deg/s, max: 3deg/s
        # self.aircraft.change_cas(2*(action[1]+1))       # max: 2 knots/s
        # self.aircraft.change_altitude(50*(action[2]+1)) # max: 50 ft/s
        
        # lat, lon, alt, head, bank, cas, fuel, _ =  self.aircraft.update()
        # distance_to_target = great_circle_distance(lat, lon, self.target_lat, self.target_lon)
        # noise_impact = self.noise_pollution_level(lat, lon, alt)
        # reward = -fuel - 10*noise_impact +\
        #             10*(self.last_distance - distance_to_target) +\
        #             (alt - self.last_alt)/50 +\
        #             (cas - self.last_cas)/2
        
        # # Update
        # self.last_alt, self.last_cas = alt, cas
        # self.last_distance = distance_to_target
        
        # dead, terminated = False, False
        # # Out of the circle
        # if not self.in_circle(lat, lon):
        #     dead, terminated = True, True 
        #     reward += 2*(10/(distance_to_target+1) - distance_to_target) - abs(alt-self.target_alt)/10 - abs(cas-self.target_cas)
        # # Check out of time
        # if self.global_time >= self.time_limit: terminated = True

        # # dump to csv
        # self.global_time += 1
        # self.save(lat, lon, alt, head, bank, cas, noise_impact)

        # return self.state_adapter(lat, lon, alt, head, bank, cas), reward, terminated, dead, (self.global_time, fuel, noise_impact)

    def save(self, lat, lon, alt, head, bank, cas, noise):
        """
        Save all states variable of one timestemp to csv file.
        """
        if self.writer:
            data = [self.global_time, (self.time + timedelta(seconds=self.global_time)).isoformat(timespec='seconds'), 0, self.aircraft.call_sign, 
                    lat, lon, alt, cas, head, bank, self.aircraft.path_angle,
                    self.aircraft.mass, self.aircraft.consumed_fuel, noise]
            
            self.writer.writerow(data)
    
    def noise_pollution_level(self, latitude, longitude, altitude, threshold=55):
        if threshold == 70:
            impact_radius = 4000
        else:
            impact_radius = 2000 * 2 ** (0.05 * (90 - threshold))

        if altitude >= impact_radius:
            return 0.0

        search_radius = (impact_radius ** 2 - altitude ** 2) ** 0.5

        nx = 728  # number of cells in x (EW) direction
        ny = 491  # number of cells in y (NS) direction
        dx = 85.714  # 62400 / 728
        dy = 92.281  # 45310 / 491
        ix = (longitude - self.population_corners['left']) / (self.population_corners['right'] - self.population_corners['left']) * nx  # position in i (has decimal)
        iy = (self.population_corners['top'] - latitude) / (self.population_corners['top'] - self.population_corners['bottom']) * ny  # position in j (has decimal)
        sx = search_radius / dx  # number of cells within radius in x direction
        sy = search_radius / dy  # number of cells within radius in y direction

        lb = min(max(round(ix - sx), 0), nx-1)  # left bound of impact region
        rb = min(max(round(ix + sx), 0), nx-1)  # right bound of impact region
        tb = min(max(round(iy - sy), 0), ny-1)  # top bound of impact region
        bb = min(max(round(iy + sy), 0), ny-1)  # bottom bound of impact region
        ixs, iys = np.meshgrid(np.arange(lb, rb + 1), np.arange(tb, bb + 1))

        impact_rectangle = self.population[tb:bb + 1, lb:rb + 1]
        impact_distance = np.sqrt(np.square(np.abs(ixs - ix) * dx) + np.square(np.abs(iys - iy) * dy) + altitude ** 2 + 1)
        impact_noise = np.square(np.maximum((70 - threshold) - 20 * np.log2(0.00025 * impact_distance), 0))

        influence = impact_noise.flatten().dot(impact_rectangle.flatten())/1e6
        return influence
          
# env = Departure()
# env.reset()
# # Heading, Acceleration (knot/s), altitude change (ft/s)
# print(env.step([0, -1, 1]))
