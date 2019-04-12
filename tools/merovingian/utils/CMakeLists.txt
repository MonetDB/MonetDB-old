#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(../../../common/stream ../../../common/utils ${CRYPTO_INCLUDE_DIR})

add_library(meroutil STATIC control.c control.h database.c database.h glob.c glob.h properties.c properties.h utils.c
			utils.h)
