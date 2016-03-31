#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <values.h>
#include <sys/time.h>
#include <time.h>

/*  Martin Kersten
 *  A minimal file to show how to create the binary files used in MonetDB
 */

/* The MonetDB types supported */
typedef signed char bte;
typedef short sht;
typedef float flt;
typedef double dbl;
typedef long lng; /* make sure it is 8 bytes */

typedef time_t clk;
typedef int date;
typedef int daytime;
typedef union {
    lng alignment;
    struct {
#ifndef WORDS_BIGENDIAN
        daytime p_msecs;
        date p_days;
#else
        date p_days;
        daytime p_msecs;
#endif
    } payload;
} timestamp;

const bte bte_nil = -127;
const sht sht_nil = SHRT_MIN;
const int int_nil = INT_MIN;
const flt flt_nil = FLT_MIN;
const dbl dbl_nil = DBL_MIN;
const lng lng_nil = LONG_MIN;
const clk time_nil = INT_MIN;
const date date_nil = INT_MIN;
const daytime daytime_nil =(daytime) INT_MIN;
const timestamp timestamp_nil =(timestamp) LONG_MIN;


int main(int argc, char **argv)
{
	int i,j,lim;
	FILE *f;

	if ( argc <3 || (argc %2 != 0) ){
		printf("use:%s <recordcount> [<filename> <type>] ...\n", argv[0]);
		return -1;
	}
	if( sizeof(lng) != 8){
		printf("only use 8-byte lng\n");
		return -1;
	}
	i = 1;
	lim = atoi(argv[i]);
	if( lim < 0){
		printf("record count <0\n");
		return -1;
	}
	
	for( i++; i< argc; i+=2){
		f= fopen(argv[i],"w");
		if( f== NULL){
			printf("could not create '%s'\n",argv[i]);
			return -1;
		}

		if( strncmp(argv[i+1],"int",3) == 0 ){
			int val= 0;
			for(j=0; j< lim; j++, val++)
				fwrite((void*)&val, sizeof(val), 1, f);
			if(strstr(argv[i+1],"nil"))
				fwrite((void*)&int_nil, sizeof(val), 1, f);

		}  else
		if( strncmp(argv[i+1],"lng",3) == 0 ){
			lng val= 0;
			for(j=0; j< lim; j++, val++)
				fwrite((void*)&val, sizeof(val), 1, f);
			if(strstr(argv[i+1],"nil"))
				fwrite((void*)&lng_nil, sizeof(val), 1, f);
		}  else
		if( strncmp(argv[i+1],"timestamp",9) == 0 ){
			timestamp val; // representation of timestamp
			val.payload.p_days=0;
			val.payload.p_msecs=0;
			for(j=0; j< lim; j++, val.payload.p_days++)
				fwrite((void*)&val, sizeof(val), 1, f);
			if(strstr(argv[i+1],"nil"))
				fwrite((void*)&timestamp_nil, sizeof(val), 1, f);
		} else
		if( strncmp(argv[i+1],"time",4) == 0 ){
			clk val;	// 	representation of date values
			ctime(&val);
			for(j=0; j< lim; j++, val++)
				fwrite((void*)&val, sizeof(val), 1, f);
			if(strstr(argv[i+1],"nil"))
				fwrite((void*)&time_nil, sizeof(val), 1, f);
		} else
		if( strncmp(argv[i+1],"date",4) == 0 ){
			date val=0;	// 	representation of date values
			for(j=0; j< lim; j++, val++)
				fwrite((void*)&val, sizeof(val), 1, f);
			if(strstr(argv[i+1],"nil"))
				fwrite((void*)&date_nil, sizeof(val), 1, f);
		} else
		if( strncmp(argv[i+1],"flt",3) == 0 ){
			flt val;
			for(j=0; j< lim; j++, val+= 1.0)
				fwrite((void*)&val, sizeof(val), 1, f);
			if(strstr(argv[i+1],"nil"))
				fwrite((void*)&flt_nil, sizeof(val), 1, f);
		} else 
		if( strncmp(argv[i+1],"dbl",3) == 0 ){
			dbl val;
			for(j=0; j< lim; j++, val+= 1.0)
				fwrite((void*)&val, sizeof(val), 1, f);
			if(strstr(argv[i+1],"nil"))
				fwrite((void*)&dbl_nil, sizeof(val), 1, f);
		} 
		else printf("invalid type %s\n", argv[i+1]);
		fclose(f);
	}
}
