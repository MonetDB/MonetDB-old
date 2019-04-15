#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

if(HAVE_SAMTOOLS)
	add_subdirectory(bam)
endif()
if(HAVE_FITS)
	add_subdirectory(fits)
endif()
if(HAVE_LIDAR)
	add_subdirectory(lidar)
endif()
if(HAVE_NETCDF)
	add_subdirectory(netcdf)
endif()
if(HAVE_SHP)
	add_subdirectory(shp)
endif()
