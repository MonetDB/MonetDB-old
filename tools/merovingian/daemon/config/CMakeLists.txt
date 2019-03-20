#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

set(Elogdir "${LOGDIR}")
set(Erundir "${RUNDIR}")
set(Ebindir "${BINDIR}")
set(Elocalstatedir "${LOCALSTATEDIR}")

configure_file(.merovingian_properties.in .merovingian_properties @ONLY)
configure_file(monetdbd.service.in monetdbd.service @ONLY)
configure_file(monetdbd.in monetdbd @ONLY)

install(FILES .merovingian_properties DESTINATION ${LOCALSTATEDIR}/monetdb5/dbfarm)
install(FILES monetdbd.service DESTINATION ${LIBDIR}/systemd/system)
install(FILES monetdbd DESTINATION ${SYSCONFDIR}/logrotate.d)
