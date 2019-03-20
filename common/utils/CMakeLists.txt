#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

include_directories(${CRYPTO_INCLUDE_DIR})

add_library(mutils OBJECT mutils.h mutils.c prompt.c mprompt.h revision.c)
set_target_properties(mutils PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(mcrypt OBJECT mcrypt.h mcrypt.c)
set_target_properties(mcrypt PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(msabaoth OBJECT muuid.h muuid.c msabaoth.h msabaoth.c)
set_target_properties(msabaoth PROPERTIES POSITION_INDEPENDENT_CODE ON)
