#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(${OPENSSL_INCS})

add_library(mutils STATIC mutils.h mutils.c prompt.c mprompt.h revision.c)
add_library(mcrypt STATIC mcrypt.h mcrypt.c)
add_library(muuid STATIC muuid.h muuid.c)
add_library(msabaoth STATIC msabaoth.h msabaoth.c)
