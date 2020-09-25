bands_use = {
    # reference: https://landsat.gsfc.nasa.gov/landsat-8/landsat-8-bands/
    # surface reflectance product: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_SR
    1: 'coastal/aerosol band, for imaging shallow water and tracking fine particles like dust and smoke',
    2: 'visible blue',
    3: 'visible green',
    4: 'visible red, used in NDVI',
    5: 'near infrared NIR, which healthy plants reflect, used in NDVI',
    6: 'shortwave infrared SWIR, for telling wet earth from dry, for different rocks and soils for geology',
    7: 'also shortwave infrared SWIR, see band 6',
    10: 'thermal infrared TIR, reports on the temperature of the ground instead of the air, useful to distinguish among clouds, irrigated vegetation, open water, natural vegetation, burnt areas, and urban heat island',
    11: 'also thermal infrared TIR, see band 10'
}

# bands that are not in the SR product:
#     8: 'panchromatic high-resolution band at 15m, collecting visible colors in one channel',
#     9: 'clouds band, helpful to locate cumulus and cirrus clouds',


# the band numbers are listed in RGB order, which PIL expects
bands_combo = {
    'visible': [4, 3, 2],
    'SWIR': [7, 5, 1],  # shortwave infrared SWIR as red, NIR as green, and deep blue as blue
}