# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.

# This script should be ran after a change has been made to a MAL signature file.

for i in *.mal 
do
	if  [ ${i}x = 'mal_init.malx' ]
	then 
		echo "SKIP $i"
		continue
	fi
	bn=`basename $i .mal`
	incl=${bn}.include
	echo ${incl}
	echo "{ \"${bn}\","  >${incl}
	cat $i | sed  -e "s/\\\\/\\\\\\\\/g" -e "/^$/d" -e "s/\"/\\\\\"/g" -e "s/.*/\"& \\\\n\"/" >>${incl}
	echo "}," >>${incl}
done

