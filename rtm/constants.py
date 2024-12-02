"""Constants (from sron_toolbox)"""

constants = {
    "gravity": 9.80665, # gravitational acceleration in m/s^2
    "gas_constant": 8.3144598, # universal gas constant in J/mol/K
    "specific_gas_constant": { # specific gas constant
        "dry_air": 287.04 # J/(kg K)
    },
    "avogadro": 6.02214076e23, # Avogadro's constant in molecules per mol
    "mass": {
      "dry_air": 0.028964, # molar mass of dry air in kg/mol (source: https://en.wikipedia.org/wiki/Density_of_air#Humidity_.28water_vapor.29)
      "water": 0.018016, # molar mass of water vapor in kg/mol
      "h2o": 0.018010565, # molar mass of H2O in kg/mol (source: HITRAN)
      "h2o18": 0.020014811, # molar mass of H2(18)O in kg/mol (source: HITRAN)
      "hdo": 0.01901674, # molar mass of HDO in kg/mol (source: HITRAN)
      "co": 0.0280101, # molar mass of CO in kg/mol 
      "no2": 0.0460055, # molar mass of NO2 in kg/mol 
      "co2": 0.04401, # molar mass of CO2 in kg/mol
      "ch4": 0.01604, # molar mass of CH4 in kg/mol
      "o3": 0.048, # molar mass of O3 in kg/mol
    },
    "reference_abundance": {
      "h2o": 0.997317, # standard abundance of H2O (source: HITRAN)
      "h2o18": 0.002000, # standard abundance of H2(18)O (source: HITRAN)
      "hdo": 3.106930e-4, # standard abundance of HDO (source: HITRAN)
      "vsmow_hdo": 3.1152e-4, # HDO reference abundance of sea water: Vienna Standard Mean Ocean Water (VSMOW)
      "vsmow_h2o18": 2.0052e-3 # H2(18)O reference abundance of sea water: Vienna Standard Mean Ocean Water (VSMOW)
    },
    "density": {
        "liquid_water": 1000.0 # density of liquid fresh water at 4 degress Celsius
    },
    "specific_heat_constant_pressure": 1007, # specific heat of air at constant pressure in J/(kg K)
    "sea_level_standard_pressure": 101325, # Pa
    "planck": 6.62607015e-34, # Planck's constant in J s
    "boltzmann": 1.380649e-23, # Blotzmann's constant in J/K
    "speed_of_light": 299792458 # speed of light in vacuum in m/s
}
constants["temperature_lapse_rate"] = constants["gravity"]/constants["specific_heat_constant_pressure"]
constants["pressure_scale_height"] = 250.0*constants["gas_constant"]/(0.029*constants["gravity"]) # pressure scale height at 250K (mean atmospheric temperature on Earth)

fillvalue_int = -2147483647 # integer fillvalue used in SICOR NetCDF files
