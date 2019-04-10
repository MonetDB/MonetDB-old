#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

install(FILES
		bat5.mal algebra.mal status.mal
		mmath.mal alarm.mal batstr.mal
		batcolor.mal batmmath.mal
		group.mal aggr.mal
		logger.mal microbenchmark.mal
		DESTINATION ${LIBDIR}/monetdb5)

if(HAVE_HGE)
	install(FILES 00_aggr_hge.mal DESTINATION ${LIBDIR}/monetdb5/autoload)
endif()
