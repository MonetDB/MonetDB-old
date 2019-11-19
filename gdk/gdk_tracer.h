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

#define DEFAULT_COMPONENT_SEL M_ALL
#define DEFAULT_LOG_LEVEL M_DEBUG
#define DEFAULT_FLUSH_LEVEL M_DEBUG

#define FILE_NAME "trace"
#define NAME_SEP '_'
#define NEW_LINE '\n'

// Print the enum as a string
#define STR(x) #x
#define ENUM_STR(x) STR(x)

// Print only the filename without the path
#define __FILENAME__ (__builtin_strrchr(__FILE__, '/') ? __builtin_strrchr(__FILE__, '/') + 1 : __FILE__)

// TODO -> Sort it per layer
// COMPONENTS 
typedef enum { 
               // ALL
               M_ALL,

               // Specific 
               ALLOC,
               PAR,
               ALGO,

               // Modules
               GEOM,
               LIDAR,
               BAM,
               FITS,
               SHP,

               // SQL
               SQL_ALL,
               SQL_ATOM_TR,
               SQL_STATISTICS,
               SQL_ORDERIDX,
               SQL_OPTIMIZER,
               SQL_WLR,
               SQL_USER,
               SQL_SCENARIO,
               SQL_CACHE_TR,
               SQL_SYMBOL,
               SQL_MVC,
               SQL_STORE,
               SQL_UPGRADES,
               SQL_RELATION,
               SQL_RESULT,
               SQL_STATEMENT,
               SQL_BAT,


               // MAL
               MAL_ALL,
               MAL_MEMO,
               MAL_SESSION,
               MAL_INTERPRETER,
               MAL_SCHEDULER,
               MAL_LISTING,
               MAL_FACTORY,
               MAL_PARSER,
               MAL_WLC,
               MAL_REDUCE,
               MAL_CLIENT,
               MAL_OIDX,
               MAL_REMOTE,
               MAL_INET,
               MAL_GROUPBY,
               MAL_TABLET,
               MAL_OLTP,
               MAL_MANIFOLD,
               MAL_RESOLVE,
               MAL_FCN,
               MAL_DATAFLOW,
               MAL_MODULE,
               MAL_SERVER,
               MAL_EXCEPTION,
               MAL_NAMESPACE,
               MAL_PROFILER,
               MAL_MAL,


               // OPT
               MAL_OPT_ALIASES,
               MAL_OPT_CANDIDATES,
               MAL_OPT_COERCION,
               MAL_OPT_COMMONTERMS,
               MAL_OPT_PIPES,
               MAL_OPT_REMAP,
               MAL_OPT_DATAFLOW,
               MAL_OPT_EVALUATE,
               MAL_OPT_INLINE,
               MAL_OPT_JIT,
               MAL_OPT_MULTIPLEX,
               MAL_OPT_REORDER,
               MAL_OPT_PROJECTIONPATH,
               MAL_OPT_WLC,
               MAL_OPT_CONSTANTS,
               MAL_OPT_COSTMODEL,
               MAL_OPT_DEADCODE,
               MAL_OPT_EMPTYBIND,
               MAL_OPT_GENERATOR,
               MAL_OPT_JSON,
               MAL_OPT_MATPACK,
               MAL_OPT_GC,
               MAL_OPT_MERGETABLE,
               MAL_OPT_MITOSIS,
               MAL_OPT_PUSHSELECT,
               MAL_OPT_QUERYLOG,
               MAL_OPT_OLTP,
               MAL_OPT_PROFILER,
               MAL_OPT_REDUCE,
               MAL_OPT_REMOTE,
               MAL_OPT_VOLCANO,
               MAL_OPT_MACRO,
               MAL_OPT_POSTFIX,
            
               // GDK
               GDK_ALL,
               GDK_LOGGER

              } COMPONENT;
              
// LOG LEVELS
typedef enum { 

               M_CRITICAL = 1,
               M_ERROR = 100, 
               M_WARNING = 150,
               M_INFO = 200,
               M_DEBUG = 255

              } LOG_LEVEL;


extern LOG_LEVEL CUR_LOG_LEVEL;

/**
 * 
 * Macros for logging
 * Function name is detected automatically
 * 
 */
#define GDK_TRACER_LOG(LOG_LEVEL, COMP, MSG, ...)                       \
    if(CUR_LOG_LEVEL >= LOG_LEVEL)                                      \
    {                                                                   \
        GDKtracer_log(LOG_LEVEL,                                        \
                      "[%s] %s <%s:%d> (%s - %s) %s # "MSG,             \
                      GDKtracer_get_timestamp("%Y-%m-%d %H:%M:%S"),     \
                      __FILENAME__,                                     \
                      __FUNCTION__,                                     \
                      __LINE__,                                         \
                      ENUM_STR(LOG_LEVEL),                              \
                      ENUM_STR(COMP),                                   \
                      MT_thread_getname(),                              \
                      ## __VA_ARGS__);                                  \
    }                                                                   \

#define CRITICAL(COMP, MSG, ...)                                              \
    GDK_TRACER_LOG(M_CRITICAL, COMP, MSG, ## __VA_ARGS__)                     \

#define ERROR(COMP, MSG, ...)                                                 \
    GDK_TRACER_LOG(M_ERROR, COMP, MSG, ## __VA_ARGS__)                        \

#define WARNING(COMP, MSG, ...)                                               \
    GDK_TRACER_LOG(M_WARNING, COMP, MSG, ## __VA_ARGS__)                      \

#define INFO(COMP, MSG, ...)                                                  \
    GDK_TRACER_LOG(M_INFO, COMP, MSG, ## __VA_ARGS__)                         \

#define DEBUG(COMP, MSG, ...)                                                 \
    GDK_TRACER_LOG(M_DEBUG, COMP, MSG, ## __VA_ARGS__)                        \


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
gdk_return GDKtracer_set_log_level(LOG_LEVEL level);


// Resets the log level to the default one - NONE. If the current log level is not NONE 
// and GDK_tracer_reset_log_level() is called we need to flush the buffer first in order to 
// "discard" messages kept from other log levels
gdk_return GDKtracer_reset_log_level(void);


// Sets the minimum flush level that an event will trigger the logger to flush the buffer
gdk_return GDKtracer_set_flush_level(LOG_LEVEL level);


// Resets the flush level to the default (ERROR)
gdk_return GDKtracer_reset_flush_level(void);


// TODO -> Write comments
// Candidate for 'gnu_printf' format attribute [-Werror=suggest-attribute=format] 
gdk_return GDKtracer_log(LOG_LEVEL level, const char *fmt, ...) __attribute__ ((format (printf, 2, 3)));


// Flush the buffer to the file. If after flushing the buffer, the file is greater 
// or equal to 1 GB, then we close the current one and we swap to a new one increasing the ID
gdk_return GDKtracer_flush_buffer(void);

#endif
