# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.

sed '/^$/q' $0			# copy copyright from this file

cat <<EOF
# This file was generated by using the script ${0##*/}.

module sql;

EOF

cat <<EOF
pattern sql.window_bound(b:any_1, unit:int, bound:int, excl:int, start:hge) :lng
address SQLwindow_bound
comment "computes window ranges for each row";

pattern batsql.window_bound(b:bat[:any_1], unit:int, bound:int, excl:int, start:hge) :bat[:lng]
address SQLwindow_bound
comment "computes window ranges for each row";

pattern sql.window_bound(p:bit, b:any_1, unit:int, bound:int, excl:int, start:hge) :lng
address SQLwindow_bound
comment "computes window ranges for each row";

pattern batsql.window_bound(p:bat[:bit], b:bat[:any_1], unit:int, bound:int, excl:int, start:hge) :bat[:lng]
address SQLwindow_bound
comment "computes window ranges for each row";

pattern batsql.window_bound(b:bat[:any_1], unit:int, bound:int, excl:int, start:bat[:hge]) :bat[:lng]
address SQLwindow_bound
comment "computes window ranges for each row";

pattern batsql.window_bound(p:bat[:bit], b:bat[:any_1], unit:int, bound:int, excl:int, start:bat[:hge]) :bat[:lng]
address SQLwindow_bound
comment "computes window ranges for each row";


EOF

for tp1 in 1:bte 2:sht 4:int 8:lng 16:hge; do
    for tp2 in 16:hge; do
	if [ ${tp1%:*} -le ${tp2%:*} -o ${tp1#*:} = ${tp2#*:} ]; then
	    cat <<EOF
pattern sql.sum(b:${tp1#*:}, s:lng, e:lng) :${tp2#*:}
address SQLsum
comment "return the sum of groups";

pattern batsql.sum(b:bat[:${tp1#*:}], s:bat[:lng], e:bat[:lng]) :bat[:${tp2#*:}]
address SQLsum
comment "return the sum of groups";

pattern sql.prod(b:${tp1#*:}, s:lng, e:lng) :${tp2#*:}
address SQLprod
comment "return the product of groups";

pattern batsql.prod(b:bat[:${tp1#*:}], s:bat[:lng], e:bat[:lng]) :bat[:${tp2#*:}]
address SQLprod
comment "return the product of groups";

EOF
	fi
    done
done

	cat <<EOF
pattern sql.avg(b:hge, s:lng, e:lng) :dbl
address SQLavg
comment "return the average of groups";

pattern batsql.avg(b:bat[:hge], s:bat[:lng], e:bat[:lng]) :bat[:dbl]
address SQLavg
comment "return the average of groups";

EOF
