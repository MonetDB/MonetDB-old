#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

install(FILES
		language.mal mal_init.mal bbp.mal
		profiler.mal batExtensions.mal orderidx.mal
		inspect.mal manual.mal mal_io.mal mkey.mal manifold.mal
		iterator.mal clients.mal
		factories.mal groupby.mal mdb.mal pcre.mal mat.mal
		transaction.mal oltp.mal wlc.mal
		mal_mapi.mal sabaoth.mal remote.mal 
		txtsim.mal
		tokenizer.mal sample.mal json_util.mal
		batmtime.mal querylog.mal sysmon.mal
		DESTINATION ${LIBDIR}/monetdb5)
install(FILES 01_calc.mal 01_batcalc.mal DESTINATION ${LIBDIR}/monetdb5/autoload)

if(HAVE_HGE)
	install(FILES
			00_calc_hge.mal 00_batcalc_hge.mal
			00_iterator_hge.mal 00_language_hge.mal
			00_mal_mapi_hge.mal 00_mkey_hge.mal 00_batExtensions_hge.mal
			DESTINATION ${LIBDIR}/monetdb5/autoload)
endif()
