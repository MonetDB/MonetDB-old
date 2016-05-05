# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.

sed '/^$/q' $0			# copy copyright from this file

integer="bte sht int lng hge"	# all integer types
numeric="$integer flt dbl"	# all numeric types

name="distance"
for tp1 in $numeric; do
	for tp2 in $numeric; do
		case $tp1$tp2 in
		*hge*)
			L1="#ifdef HAVE_HGE"
			L2="#endif"
		;;
		*)
			L1=""
			L2=""
		esac
		cat <<EOF
${L1}
GRID_dist(${tp1},${tp2});
${L2}
EOF
	done
done
echo
