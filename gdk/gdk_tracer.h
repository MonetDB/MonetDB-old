/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef _GDK_TRACER_H_
#define _GDK_TRACER_H_

#define INT_MAX_LEN ((__CHAR_BIT__ * sizeof(int) - 1) / 3  + 2)
#define BUFFER_SIZE 64000

#define DEFAULT_LOG_LEVEL M_INFO
#define DEFAULT_FLUSH_LEVEL M_ERROR

#define FILE_NAME "trace"
#define NAME_SEP '_'
#define NEW_LINE '\n'
#define MAX_FILE_SIZE 1073741824

#define Trace(LVL, MSG, ...)                                        \
    GDKtracer_log(LVL,                                              \
                  "[%s] %s (%s:%d) # "MSG,                          \
                  GDKtracer_get_timestamp("%Y-%m-%d %H:%M:%S"),     \
                  __FILE__,                                         \
                  __FUNCTION__,                                     \
                  __LINE__,                                         \
                  ## __VA_ARGS__);


#define TraceLN(LVL, FUNC_LN, MSG, ...)                             \
    GDKtracer_log(LVL,                                              \
                  "[%s] %s (%s:%d) # "MSG,                          \
                  GDKtracer_get_timestamp("%Y-%m-%d %H:%M:%S"),     \
                  __FILE__,                                         \
                  FUNC_LN,                                          \
                  __LINE__,                                         \
                  ## __VA_ARGS__);


/**
 *
 *      NONE        0 
 *      INFO        1 - 99          Reserved for package specific debugging
 *      WARNING     100 - 149     
 *      DEBUG       150 - 199       Following the DEBUG settings in the source code
 *      ERROR       200 - 254       Package specific errors (GDK, MAL, SQL, MAPI)
 *      CRITICAL    255             Cannot be ignored
 * 
 */
// The minimum code is assigned to each LOG_LEVEL
typedef enum { 
    
               M_NONE = 0,
               M_INFO = 1, 
               M_WARNING = 100, 
               M_DEBUG = 150, 
               M_ERROR = 200, 
               M_CRITICAL = 255 

              } LOG_LEVEL;


// GDKtracer Buffer
typedef struct GDKtracer
{
    int id;
    char buffer[BUFFER_SIZE];
    int allocated_size;
    MT_Lock lock;
}
gdk_tracer;



/**
 *  GDKtracer API
 */
// Returns the timestamp in the form of datetime
char* GDKtracer_get_timestamp(char* fmt);


// Initialize tracer - basically creates the file
gdk_return GDKtracer_init(void);


// Flushes the contents of the buffer and closes the log file
gdk_return GDKtracer_stop(void);


// Sets the log level to one of the enum LOG_LEVELS above. If the current log level 
// is not NONE and GDK_tracer_set_log_level sets it to NONE we flush the buffer first 
// in order to "discard" the messages that are there from the previous log levels
gdk_return GDKtracer_set_log_level(int *level);


// Resets the log level to the default one - NONE. If the current log level is not NONE 
// and GDK_tracer_reset_log_level() is called we need to flush the buffer first in order to 
// "discard" messages kept from other log levels
gdk_return GDKtracer_reset_log_level(void);


// Sets the minimum flush level that an event will trigger the logger to flush the buffer
gdk_return GDKtracer_set_flush_level(int *level);


// Resets the flush level to the default (ERROR)
gdk_return GDKtracer_reset_flush_level(void);


// TODO -> Write comments
// Candidate for 'gnu_printf' format attribute [-Werror=suggest-attribute=format] 
gdk_return GDKtracer_log(LOG_LEVEL level, const char *fmt, ...) __attribute__ ((format (printf, 2, 3)));


// Flush the buffer to the file. If after flushing the buffer, the file is greater 
// or equal to 1 GB, then we close the current one and we swap to a new one increasing the ID
gdk_return GDKtracer_flush_buffer(void);

#endif
