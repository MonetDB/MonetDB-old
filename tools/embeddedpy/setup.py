from distutils.core import setup, Extension
import numpy.distutils.misc_util
import ntpath
import os
import sys

package_name = "monetdblite" #if you change this you also need to change the init function in embedded_module.c

if os.name == 'nt':
    print("Windows is not currently supported.")
    exit(1)

#to get the path where the python library is build we build a temporary package (maybe there's a better way to do this)
dist = setup(name="temporary_package")

#look through the output files of the temporary package to find the directory
output_directory = None
for file in dist.get_command_obj('install').get_outputs():
    if file.endswith('.egg-info'):
        output_directory = os.path.join(ntpath.dirname(file), 'monetdb5')
        break

if output_directory == None:
    print("Could not find output directory")
    exit(1)

#remove all files made by the temporary package
for file in dist.get_command_obj('install').get_outputs():
    os.remove(file)

#now actually built the package
base_dir = '../../' #relative path to the basedir of MonetDB
#include directories required
subdirs = ['', 'clients/mapilib', 'common/options', 'common/stream', 'common/utils', 'gdk', 'monetdb5', 'monetdb5/mal',
'monetdb5/modules/atoms', 'monetdb5/modules/mal', 'sql/backends/monet5', 'sql/common', 'sql/include', 'sql/server',  'sql/storage',]

#use monetdbs build system to build and distribute the .so and .mal files
current_dir = os.getcwd()
os.chdir(base_dir)
if os.system('./bootstrap') != 0:
    print("Failed bootstrapping MonetDB.")
    exit(1)
if os.system('./configure --prefix=' + output_directory + ' --enable-embedded=true --disable-fits --disable-geom --disable-rintegration --disable-gsl --disable-netcdf --disable-jdbc --disable-merocontrol --disable-odbc --disable-console --disable-microhttpd --without-perl --without-python2 --without-python3 --without-rubygem --without-unixodbc --without-samtools --without-sphinxclient --without-geos --without-samtools --without-readline --enable-silent-rules') != 0:
    print("Failed configuring MonetDB.")
    exit(1)
if os.system('make clean install') != 0:
    print("Failed building MonetDB.")
    exit(1)
os.chdir(current_dir)
#build libpyembedded.so
if os.system('make clean install') != 0:
    print("Failed building MonetDB.")
    exit(1)

#now actually create the package, the package only statically links to libpyembedded.so
setup(
    name=package_name,
    version='1.0',
    description='Embedded MonetDB Python client.',
    author='Mark Raasveldt',
    ext_modules=[Extension(
        name=package_name,
        sources=['embedded_module.c'],
        depends=['embeddedpy.h'],
        libraries=['_sql', '_pyapi'],
        library_dirs=[os.path.join(output_directory, 'lib'), os.path.join(output_directory, 'lib/monetdb5')],
        include_dirs=[os.path.join(base_dir, x) for x in subdirs],
        extra_objects = [os.path.join(output_directory, 'lib/libpyembedded.so')],
        runtime_library_dirs=[os.path.join(output_directory, 'lib/'), os.path.join(output_directory, 'lib/monetdb5')])])

