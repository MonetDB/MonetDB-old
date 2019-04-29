#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

# Readline libraries
set(READLINE_LIBRARIES "")
if(NOT ${WITH_READLINE} STREQUAL "NO")
	cmake_push_check_state()
	set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};readline")
	check_symbol_exists("readline" "stdio.h;readline/readline.h;readline/history.h" HAVE_READLINE)
	if(HAVE_READLINE)
		list(APPEND READLINE_LIBRARIES readline)
	else()
		set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};termcap")
		check_symbol_exists("rl_history_search_forward" "stdio.h;readline/readline.h;readline/history.h" HAVE_RL_HISTORY_SEARCH_FORWARD)
		if(HAVE_RL_HISTORY_SEARCH_FORWARD)
			list(APPEND READLINE_LIBRARIES termcap)
		else()
			cmake_pop_check_state()
			cmake_push_check_state()
			set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};readline;ncurses")
			check_symbol_exists("rl_reverse_search_history" "stdio.h;readline/readline.h;readline/history.h" HAVE_RL_REVERSE_SEARCH_HISTORY)
			if(HAVE_RL_REVERSE_SEARCH_HISTORY)
				list(APPEND READLINE_LIBRARIES ncurses)
			elseif(${WITH_READLINE} STREQUAL "YES")
				message(FATAL_ERROR "Readline library not found")
			else()
				message(STATUS "Readline library not found")
			endif()
		endif()
	endif()
	if(HAVE_READLINE OR HAVE_RL_HISTORY_SEARCH_FORWARD OR HAVE_RL_REVERSE_SEARCH_HISTORY)
		check_symbol_exists("rl_completion_matches" "stdio.h;readline/readline.h" HAVE_RL_COMPLETION_MATCHES)
		if(HAVE_RL_COMPLETION_MATCHES)
			check_c_source_compiles("
				#include <stdio.h>
				#include <readline/readline.h>

				int main(int argc, char **argv) {
					rl_completion_func_t *func = NULL;
					(void) argc;(void) argv;(void) func;
					return 0;
				}" HAVE_RL_COMPLETION_FUNC_T)
			if(HAVE_RL_COMPLETION_FUNC_T)
				set(HAVE_LIBREADLINE ON CACHE INTERNAL "readline library is available")
			elseif(${WITH_READLINE} STREQUAL "YES")
				message(FATAL_ERROR "readline/readline.h does not contain rl_completion_func_t, is it GNU readline?")
			else()
				message(STATUS "readline/readline.h does not contain rl_completion_func_t, is it GNU readline?")
			endif()
		elseif(${WITH_READLINE} STREQUAL "YES")
			message(FATAL_ERROR "Readline library does not contain rl_completion_matches")
		else()
			message(STATUS "Readline library does not contain rl_completion_matches")
		endif()
	endif()
	cmake_pop_check_state()
endif()

include_directories(../mapilib ../../common/options ../../common/stream ../../common/utils ${ICONV_INCLUDE_DIR})

add_library(mcutil STATIC dump.c dotmonetdb.c dotmonetdb.h eventparser.c eventparser.h)

add_executable(mclient mclient.c ReadlineTools.c ReadlineTools.h mhelp.c mhelp.h)
target_link_libraries(mclient PRIVATE mcutil mutils mcrypt moptions mapi stream ${READLINE_LIBRARIES}
					  ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES}
					  ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES} ${ICONV_LIBRARIES} ${CRYPTO_LIBRARIES})

add_executable(msqldump msqldump.c msqldump.h)
target_link_libraries(msqldump PRIVATE mcutil mutils moptions mapi stream ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES}
					  ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES}
					  ${ICONV_LIBRARIES} ${CRYPTO_LIBRARIES})

add_executable(stethoscope stethoscope.c)
target_link_libraries(stethoscope PRIVATE mcutil mutils moptions mapi stream ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES}
					  ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES}
					  ${ICONV_LIBRARIES} ${CRYPTO_LIBRARIES})

add_executable(tachograph tachograph.c)
target_link_libraries(tachograph PRIVATE mcutil mutils moptions mapi stream ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES}
					  ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES}
					  ${ICONV_LIBRARIES} ${CRYPTO_LIBRARIES})

add_executable(tomograph tomograph.c)
target_link_libraries(tomograph PRIVATE mcutil mutils moptions mapi stream ${SOCKET_LIBRARIES} ${ZLIB_LIBRARIES}
					  ${BZIP2_LIBRARIES} ${SNAPPY_LIBRARIES} ${LZ4_LIBRARIES} ${LIBLZMA_LIBRARIES} ${CURL_LIBRARIES}
					  ${ICONV_LIBRARIES} ${CRYPTO_LIBRARIES})

install(TARGETS mclient msqldump stethoscope tachograph tomograph DESTINATION ${BINDIR})
if(WIN32)
	install(FILES $<TARGET_PDB_FILE:mclient> $<TARGET_PDB_FILE:msqldump> $<TARGET_PDB_FILE:stethoscope>
			$<TARGET_PDB_FILE:tachograph> $<TARGET_PDB_FILE:tomograph> DESTINATION ${BINDIR} OPTIONAL)
else()
	install(FILES mclient.1 msqldump.1 DESTINATION ${MANDIR}/man1)
endif()
