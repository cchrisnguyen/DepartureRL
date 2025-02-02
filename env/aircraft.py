import numpy as np

from .bada import Bada
from .utils.unit_conversion import Unit
from .utils.calculation import Cal
from .utils.enums import PerformanceModel


class Aircraft:
    def __init__(self, call_sign, aircraft_type, lat, lon, alt, 
                 heading, path_angle, bank_angle, cas, vs, config, fuel_weight, payload_weight,
                 performance_model=PerformanceModel.BADA, max_cas=350, max_alt=20000):
        
        """
        Initialize the Aircraft object.
        
        Parameters:
        - call_sign (str): The call sign of the aircraft
        - aircraft_type (str): The type of the aircraft
        - lat (float): Initial latitude
        - lon (float): Initial longitude
        - alt (float): Initial altitude in feet
        - heading (float): Initial heading in degrees
        - path_angle (float): Initial flight path angle in degrees
        - bank_angle (float): Initial bank angle in degrees
        - cas (float): Initial calibrated airspeed (CAS) in knots
        - vs (float): Initial vertical speed in feet per minute
        - config (str): Aircraft configuration
        - fuel_weight (float): Initial fuel weight in kilograms
        - payload_weight (float): Initial payload weight in kilograms
        """

        if performance_model not in PerformanceModel._value2member_map_: raise Exception("Performance model unknown!")
        self.perf = Bada(aircraft_type)
        self.G0 = self.perf.get_g_0()

        self.call_sign = call_sign
        self.aircraft_type = aircraft_type
        self.lat = lat
        self.lon = lon
        self.alt = Unit.ft2m(alt)
        self.heading = heading
        self.path_angle = path_angle
        self.bank_angle = bank_angle
        self.cas = Unit.kts2mps(cas)
        self.vs = Unit.ftpm2mps(vs)
        self.config = config
        self.fuel = fuel_weight
        self.consumed_fuel = 0
        self.mass =  self.perf.get_empty_weight() + payload_weight + fuel_weight
        
        self.d_T = 0
        self.T = self.perf.cal_temperature(Unit.ft2m(alt), self.d_T)
        self.p = self.perf.cal_air_pressure(Unit.ft2m(alt), self.T, self.d_T)
        self.rho = self.perf.cal_air_density(self.p, self.T)

        self.tas = self.perf.cas_to_tas(self.cas, self.p, self.rho)
        self.rate_of_turn = self.perf.cal_rate_of_turn(self.bank_angle, self.tas)

        self.max_cas = Unit.kts2mps(max_cas)
        self.max_alt = Unit.ft2m(max_alt)
        self.max_bank_angle = self.perf.get_norminal_bank_angle()
        #----------------------------- Autopilot -------------------------
        self.ap_heading = self.heading
        self.ap_cas = self.cas
        self.ap_alt = self.alt
        self.failure = False

    def reset(self, lat, lon, alt, 
                heading, path_angle, bank_angle, 
                cas, vs, config, fuel_weight, payload_weight):
        
        """
        Reset the aircraft's state to the initial conditions.
        
        Parameters:
        - lat (float): Initial latitude
        - lon (float): Initial longitude
        - alt (float): Initial altitude in feet
        - heading (float): Initial heading in degrees
        - path_angle (float): Initial flight path angle in degrees
        - bank_angle (float): Initial bank angle in degrees
        - cas (float): Initial calibrated airspeed (CAS) in knots
        - vs (float): Initial vertical speed in feet per minute
        - fuel_weight (float): Initial fuel weight in kilograms
        - payload_weight (float): Initial payload weight in kilograms
        """

        self.lat = lat
        self.lon = lon
        self.alt = Unit.ft2m(alt)
        self.heading = heading
        self.path_angle = path_angle
        self.bank_angle = bank_angle
        self.cas = Unit.kts2mps(cas)
        self.vs = Unit.ftpm2mps(vs)
        self.config = config
        self.fuel = fuel_weight
        self.consumed_fuel = 0
        self.mass =  self.perf.get_empty_weight() + payload_weight + fuel_weight
        
        self.d_T = 0
        self.T = self.perf.cal_temperature(Unit.ft2m(alt), self.d_T)
        self.p = self.perf.cal_air_pressure(Unit.ft2m(alt), self.T, self.d_T)
        self.rho = self.perf.cal_air_density(self.p, self.T)

        self.tas = self.perf.cas_to_tas(self.cas, self.p, self.rho)
        self.rate_of_turn = self.perf.cal_rate_of_turn(self.bank_angle, self.tas)
        #----------------------------- Autopilot -------------------------
        self.ap_heading = self.heading
        self.ap_cas = self.cas
        self.ap_alt = self.alt
        self.failure = False

    def change_heading(self, delta):
        """
        Change the aircraft's heading.
        
        Parameters:
        - delta (float): The change in heading in degrees
        """
        self.ap_heading = self.heading + delta

    def change_cas(self, delta):
        """
        Change the aircraft's calibrated airspeed (CAS).
        
        Parameters:
        - delta (float): The change in CAS in knots
        """
        self.ap_cas = min(self.max_cas, self.cas + Unit.kts2mps(delta))

    def change_altitude(self, delta):
        """
        Change the aircraft's altitude.
        
        Parameters:
        - delta (float): The change in altitude in feet
        """
        self.ap_alt = min(self.max_alt, self.alt + Unit.ft2m(delta))

    def update(self):
        """
        Update the aircraft's state, including atmosphere and bank angle.

        This function updates the aircraft's atmospheric conditions such as temperature, pressure, 
        and air density based on the current altitude. It also calculates the required bank angle 
        to achieve the desired heading and updates the aircraft's bank angle accordingly.
        """
        # Update atmosphere
        self.d_T = 0 #self.weather.get_dT(self.lat, self.lon, self.alt)
        self.T = self.perf.cal_temperature(self.alt, self.d_T)
        self.p = self.perf.cal_air_pressure(self.alt, self.T, self.d_T)
        self.rho = self.perf.cal_air_density(self.p, self.T)

        # Bank angle
        r_rate_of_turn = Cal.cal_angle_diff(self.heading, self.ap_heading)
        r_bank_angle = self.perf.cal_bank_angle(r_rate_of_turn, self.tas)
        self.bank_angle += max(min(r_bank_angle - self.bank_angle, 5),-5) # Choose 5 deg/s as the maximum bank angle rate
        self.bank_angle = max(min(self.bank_angle, self.max_bank_angle),-self.max_bank_angle)
        self.rate_of_turn = self.perf.cal_rate_of_turn(self.bank_angle, self.tas)

        self.heading += self.rate_of_turn
        self.heading = (self.heading+360) % 360
        
        # Change speed
        r_tas = self.perf.cas_to_tas(self.ap_cas, self.p, self.rho)
        r_dVdt = r_tas - self.tas
        r_dhdt = self.ap_alt - self.alt

        drag = self.perf.cal_aerodynamic_drag(self.tas, self.vs, self.bank_angle, self.mass, self.rho, self.config, 1.0)
        h_max = self.perf.cal_maximum_altitude(self.d_T, self.mass)
        max_thrust = 2*self.perf.cal_max_climb_to_thrust(Unit.m2ft(self.alt), Unit.mps2kts(self.tas), self.d_T)*self.perf.cal_reduced_climb_power(self.mass, self.alt, h_max)
        for _ in range(10):
            r_thrust = drag + self.mass*(self.G0*r_dhdt/self.tas +r_dVdt)
            thrust = min(r_thrust, max_thrust)
            r_dhdt = thrust/(r_thrust + 10)*r_dhdt
            r_dVdt = thrust/(r_thrust + 10)*r_dVdt
            if r_thrust <= max_thrust: break

        self.vs = r_dhdt
        self.tas += r_dVdt
        self.cas = self.perf.tas_to_cas(self.tas, self.p, self.rho)
        if self.cas < Unit.kts2mps(self.perf.get_v_stall(self.config)):
            self.failure = True #!!! STALL !!!#
        self.path_angle = np.arcsin(self.vs / self.tas)

        # Ground speed
        gs_true = self.tas * np.cos(self.path_angle)
        self.gs_north =  gs_true * np.cos(np.deg2rad(self.heading))
        self.gs_east =  gs_true * np.sin(np.deg2rad(self.heading))
        
        # Position
        self.lat = self.lat + self.gs_north / 110728
        self.lon = self.lon + self.gs_east / 103262
        self.alt = self.alt + self.vs
        if self.alt <= 1000:
            self.failure = True

        # Fuel
        fuel_burn = self.perf.cal_fuel_burn(self.config, self.tas, thrust, self.alt)
        self.consumed_fuel += fuel_burn
        self.fuel -= fuel_burn
        self.mass -= fuel_burn

        return self.lat, self.lon, Unit.m2ft(self.alt), self.heading, self.bank_angle, Unit.mps2kts(self.cas), fuel_burn, r_thrust
    