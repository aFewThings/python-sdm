README file for rasters of environmental data
These rasters are part of the data publication: Elith, J., Graham, C.H., Valavi, R., Abegg, M., Bruce, C., Ferrier, S., Ford, A., Guisan, A., Hijmans, R.J., Huettmann, F., Lohmann, L.G., Loiselle, B.A., Moritz, C., Overton, J.McC., Peterson, A.T., Phillips, S., Richardson, K., Williams, S., Wiser, S.K., Wohlgemuth, T. & Zimmermann, N.E. (2020) Presence-only and presence-absence data for comparing species distribution modeling methods. Biodiversity Informatics 15:69-80

Rasters are arranged in folders by region, and presented as .tif files. 
Each region has a metadata file describing the variables; this current file adds authors responsible for data preparation and details of coordinate reference systems, units and raster cell sizes.

Projections are displayed as per the PROJ.4 library available in R (used by the packages 'rgdal' and 'sp'): https://proj4.org/

# AWT
Prepared by Karen Richardson, Caroline Bruce, Catherine Graham
13 variables:  see AWT\01_metadata_AWT_environment.csv
Coordinate reference system: UTM, zone 55, spheroid GRS 1980, datum GDA94
EPSG:28355
Units: m
Raster cell size: 80 m

#CAN
Prepared by Falk Huettmann, Jane Elith and Catherine Graham
11 variables:  see CAN\01_metadata_CAN_environment.csv
Coordinate reference system: unprojected, Clarke 1866 ellipsoid
EPSG:4008
Units: decimal degree
Raster cell size: 0.008333334  degrees (~ 1km)

#NSW
Provided by Simon Ferrier
13 variables:  see NSW\01_metadata_NSW_environment.csv
Coordinate reference system: unprojected WGS84 datum
EPSG:4326
Units: decimal degree
Raster cell size: 0.000899322 degrees (approx 100m)

#NZ
Provided by Jake Overton 
13 variables:  see NZ\01_metadata_NZ_environment.csv
New Zealand Map Grid (NZMG), Datum: NZGD49 (New Zealand Geodetic Datum 1949), Ellipsoid: International 1924
EPSG:27200
Units: meters
Raster cell size 100m

#SA
Prepared by Bette Loiselle, Lucia Lohmann and Catherine Graham
11 variables:  see SA\01_metadata_SA_environment.csv
Coordinate reference system: unprojected, WGS84 datum
EPSG:4326
Units: decimal degree
Raster cell size: 0.008333333 degrees (~ 1km)


#SWI
Prepared by Niklaus E. Zimmermann and Antoine Guisan
13 variables:  see SWI\01_metadata_SWI_environment.csv
Coordinate reference system: Transverse, spheroid Bessel (note all data has had a constant shift applied to it)
EPSG:21781
Units: meters
Raster cell size: 100m


