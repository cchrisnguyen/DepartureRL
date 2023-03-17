import numpy as np
import xarray as xr
import cdsapi
from datetime import datetime, time
from pathlib import Path

from .utils.unit_conversion import Unit
from .utils.enums import WeatherMode


class Weather:
    def __init__(self, weather_mode, time):
        self.mode = weather_mode
        if self.mode == WeatherMode.ONLY_CRUISE_WIND:
            cruising_wind = self.download_cruising_wind(time)
            self.wind_data = xr.open_dataset(cruising_wind)

    def change_time(self, time):
        if self.mode == WeatherMode.ONLY_CRUISE_WIND:
            cruising_wind = self.download_cruising_wind(time)
            self.wind_data = xr.open_dataset(cruising_wind)

    def get_dT(self, lat, lon, alt):
        if self.mode == WeatherMode.WEATHER:
            raise Exception("Weather has not been implemented yet!")
        return 0
    
    def get_wind(self, lat, lon, alt):
        if self.mode == WeatherMode.WEATHER:
            raise Exception("Weather has not been implemented yet!")
        
        if self.mode == WeatherMode.ONLY_CRUISE_WIND:
            ds = self.wind_data.interp(longitude=xr.DataArray(lon, dims="z"), latitude=xr.DataArray(lat, dims="z"))
            wind_east = Unit.mps2kts(np.array(ds['u'].values))
            wind_north = Unit.mps2kts(np.array(ds['v'].values))
            return wind_east, wind_north
        
        return 0, 0
    
    @staticmethod
    def download_cruising_wind(start_time):
        c = cdsapi.Client()
        if not Path(__file__).parent.parent.parent.parent.resolve().joinpath(f'data/weather/cruising'):
            Path(__file__).parent.parent.parent.parent.resolve().joinpath(f'data/weather/cruising').mkdir()

        if not Path(__file__).parent.parent.parent.parent.resolve().joinpath(f'data/weather/cruising/{start_time.strftime("%Y-%m-%d")}.nc').exists():
            print("Downloading ERA5 data.")
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': ['u_component_of_wind', 'v_component_of_wind',],
                    'pressure_level': [250,],
                    'year': start_time.year,
                    'month': start_time.month,
                    'day': start_time.day,
                    'time': [time(hour=hour).isoformat(timespec='minutes') for hour in range(24)],
                    'format': 'netcdf',
                },
                Path(__file__).parent.parent.parent.parent.resolve().joinpath(f'data/weather/cruising/{start_time.strftime("%Y-%m-%d")}.nc'))
        
        return Path(__file__).parent.parent.parent.parent.resolve().joinpath(f'data/weather/cruising/{start_time.strftime("%Y-%m-%d")}.nc')

    @staticmethod
    def get_cruising_wind_field(time_list):
        data = []
        for time in time_list:
            time = datetime.fromtimestamp(time)
            data.append(Weather.cruising_wind(time))

        return data

    @staticmethod
    def cruising_wind(time):
        cruising_wind = Weather.download_cruising_wind(time)
        wind_data = xr.open_dataset(cruising_wind)
        sliced_wind_data = wind_data.sel(longitude=slice(114.5,121), latitude=slice(25,21.5), time=np.datetime64((time).replace(second=0, minute=0),'ns')).to_array().to_numpy()/200 # DIVIDE BY 200!!!!
        return sliced_wind_data