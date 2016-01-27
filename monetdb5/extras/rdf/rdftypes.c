/*
 * The contents of this file are subject to the MonetDB Public License
 * Version 1.1 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 * http://www.monetdb.org/Legal/MonetDBLicense
 *
 * Software distributed under the License is distributed on an "AS IS"
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
 * License for the specific language governing rights and limitations
 * under the License.
 *
 * The Original Code is the MonetDB Database System.
 *
 * The Initial Developer of the Original Code is CWI.
 * Portions created by CWI are Copyright (C) 1997-July 2008 CWI.
 * Copyright August 2008-2013 MonetDB B.V.
 * All Rights Reserved.
 */

/*
 * @a Minh-Duc Pham
 *
 */
#include "monetdb_config.h"
#include "mal_exception.h"
#include <gdk.h>
#include <rdf.h>
#include <rdftypes.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctype.h>

/*
 * Get substring of a string
 * NOTE: The position starts from 1 (not from 0)
 * */

char* substring(char *string, int position, int length) 
{
	char *pointer;
	int c;

	pointer = GDKmalloc(length+1);

	if (pointer == NULL)
	{
		throw(RDF, "rdf_shredder.substring", "Memory allocation failed!");
	}

	for (c = 0 ; c < position -1 ; c++) 
		string++; 

	for (c = 0 ; c < length ; c++)
	{
		*(pointer+c) = *string;      
		string++;   
	}

	*(pointer+c) = '\0';

	return pointer;
}


/* 
	This function returns 1 if architecture 
	is little endian, 0 in case of big endian.
*/

int is_little_endian(void)
{
	unsigned int x = 1;
	char *c = (char*) &x;
	return (int)*c;
}

char isInt(char *input, int len){
	
	int 	i = 0;
	//int	len = strlen(input);
	//printf("... Checking value %s with len %d \n", input, len);
	if (len > 11) return 0;

	if (input[0] != '-' && input[0] != '+' && isdigit(input[0]) == 0)
		return 0; 	

	for(i = 1; i < len; i++)
	{
		if(isdigit(input[i]) == 0){ // May also check ispunct(string[i]) != 0 
			//printf("NOT A DIGIT \n");
			break;
		}
	}
	//printf("i is %d \n",i);
	if(i == len)
		return 1;
	else
		return 0;
}

char isDouble(char *input, int len){
	
	int 	i = 0;
	int	numE = 0; 	//number of E's
	//int	len = strlen(input);
	//printf("... Checking value %s with len %d \n", input, len);
	if (input[0] != '-' && input[0] != '+' && isdigit(input[0]) == 0)
		return 0; 	

	for(i = 1; i < len; i++)
	{
		if(isdigit(input[i]) == 0 && input[i] != '.' 
		   && input[i] != 'e' && input[i] != 'E' 
		   && input[i] != '-' && input[i] != '+'){ // May also check ispunct(string[i]) != 0 
			//printf("NOT A DIGIT \n");
			break;
		}
		if (input[i] == 'e' || input[i] == 'E')
			numE++;
	}

	if (numE > 1) return 0; 

	//printf("i is %d \n",i);
	if(i == len)
		return 1;
	else
		return 0;
}

int getIntFromRDFString(str input){
	int i; 
	int ret; 
	char *tmpStr; 

	assert(input[0] == '\"');
	
	//Find the second quote
	i = 1; 
	while (input[i] != '\"'){
		i++; 
	}
	//input[i] = '\0';
	//input++;
	tmpStr = substring(input, 2, i - 1);
	//printf("INT: Input after extraction %s \n", tmpStr);	
	ret = atoi(tmpStr);
	//printf("return value: %d \n",ret); 
	GDKfree(tmpStr); 
	return ret; 
}

double getDoubleFromRDFString(str input){
	int i; 
	double ret; 
	char *tmpStr; 

	assert(input[0] == '\"');
	
	//Find the second quote
	i = 1; 
	while (input[i] != '\"'){
		i++; 
	}
	tmpStr = substring(input, 2, i - 1);
	ret = atof(tmpStr); 
	//printf("return value: %f \n",ret); 
	GDKfree(tmpStr);
	return ret; 
}

str getDateTimeFromRDFString(str input){

	int i; 
	char *tmpStr; 
	assert(input[0] == '\"');
	
	//Find the second quote
	i = 1; 
	while (input[i] != '\"'){
		i++; 
	}
	tmpStr = substring(input, 2, i - 1);

	return tmpStr; 
}

static 
char rdfcastToIntFromDouble(ValPtr srcPtr, ValPtr dstPtr){
	double srcdbl; 
	int  dstint; 
	srcdbl =  srcPtr->val.dval;
	dstint = (int) srcdbl; 
	if (dstint == srcdbl){
		VALset(dstPtr,TYPE_int, &dstint);		
		return 1; 		
	}
	else return 2; 
}



/*
 * Note: Input is a quoted string 
 * Get the first quoted part of the string 
 * e.g., "abc"@en ===>  abc
 * Check whether it can be converted to an integer or not
 * */

static 
char rdfcastToIntFromString(ValPtr srcPtr, ValPtr dstPtr){
	int i; 
	int dstint; 
	char *tmpStr; 
	int const MAX_INT_LEN = 9; 
	char *input = srcPtr->val.sval; 

	assert(input[0] == '\"');

	if (srcPtr->len < 3) return 0;
	
	if (input[1] != '-' && isdigit(input[1]) == 0)
		return 0; 	

	//Find the second quote
	i = 2; 
	while (input[i] != '\"'){
		if (isdigit(input[i]) == 0) return 0;
		i++; 
	}

	if (i > (MAX_INT_LEN + 1)) return 0;

	tmpStr = substring(input, 2, i - 1);
	//printf("INT: Input after extraction %s \n", tmpStr);	
	dstint = atoi(tmpStr);
	GDKfree(tmpStr); 
	VALset(dstPtr,TYPE_int, &dstint);
	return 1; 
}


static 
char rdfcastToDoubleFromString(ValPtr srcPtr, ValPtr dstPtr){
	int i; 
	int	numdot = 0; 
	double dstdbl; 
	char *tmpStr; 
	int const MAX_DOUBLE_LEN = 23; 
	char *input = srcPtr->val.sval; 
	
	assert(input[0] == '\"');

	if (srcPtr->len < 3) return 0;
	
	if (input[1] != '-' && isdigit(input[1]) == 0)
		return 0; 	

	//if (ispunct(input[1])) return 0; 
	if (ispunct(input[1]) == '.') return 0; 

	//Find the second quote
	i = 2; 
	while (input[i] != '\"'){
		//if (isdigit(input[i]) == 0 && !ispunct(input[i])) return 0;
		if (isdigit(input[i]) == 0 && (input[i] != '.' )) 
			return 0;

		//if (ispunct(input[i])) numdot += 1; 
		if (input[i] == '.') numdot += 1; 

		i++; 
	}
	
	if (numdot > 1)	return 0; 

	if (i > (MAX_DOUBLE_LEN + 1)) return 0;

	tmpStr = substring(input, 2, i - 1);
	dstdbl = atof(tmpStr);
	//printf("Float: Input after extraction %s ==> %f \n", tmpStr, dstdbl);	
	GDKfree(tmpStr); 
	VALset(dstPtr,TYPE_dbl, &dstdbl);
	return 1; 
}

/**
 * Castable Table
 * 	src:	URI	STRING	DATETIME	DOUBLE	INT
 * dst: 	
 * URI				
 * STRING		Y	Y		Y	Y	
 * DATETIME			
 * DOUBLE		V				Y
 * INT			V			V	
 *
 * Y: Castable
 * V: need to be verified
 * */

 /* Return: 
 * 0: Cannot cast
 * 1: Cast without changing the value
 * 2: Castable, but the dest value is not the same as the src value 
 */


char rdfcast(ObjectType srcT, ObjectType dstT, ValPtr srcPtr, ValPtr dstPtr){
	switch(dstT){
		case URI: 
			return 0; 	// Do not convert any thing to URI
		case STRING: 
			switch(srcT){
				case INTEGER:
					dstPtr->val.sval = NULL;
					(*BATatoms[TYPE_int].atomToStr)(&dstPtr->val.sval,
							&dstPtr->len, &srcPtr->val.ival);
					dstPtr->vtype = TYPE_str;
					//printf("Cast a int value %d to string value %s  len = %d \n", srcPtr->val.ival, dstPtr->val.sval, dstPtr->len); 
					return 1; 
				case DOUBLE: 					
					dstPtr->val.sval = NULL;
					(*BATatoms[TYPE_dbl].atomToStr)(&dstPtr->val.sval,
							&dstPtr->len, &srcPtr->val.dval);
					dstPtr->vtype = TYPE_str;
					return 1; 
				case DATETIME: 	//Datetime in encoded lng value of timestamp
					{	
					char buf[64], *s1 = buf;
					int len = 64; 
					timestamp ts; 
					convert_encodedLng_toTimestamp(srcPtr->val.lval, &ts);
					
					*s1 = 0;
					timestamp_tostr(&s1,&len,&ts);

					dstPtr->vtype = TYPE_str; 
					dstPtr->len = srcPtr->len; 
					dstPtr->val.sval = GDKstrdup(s1);
					}
					return 1; 
				case STRING:	
					dstPtr->vtype = TYPE_str; 
					dstPtr->len = srcPtr->len; 
					dstPtr->val.sval = GDKstrdup(srcPtr->val.sval);
					return 1; 
				default: 
					return 0; 
			}
		case DATETIME: 
			return 0; 
		case DOUBLE: 
			switch(srcT){
				case INTEGER: 
					dstPtr->vtype = TYPE_dbl; 
					dstPtr->val.dval = (double) srcPtr->val.ival;
					return 1; 
				case STRING: 
					return rdfcastToDoubleFromString(srcPtr, dstPtr);
				default: 
					return 0; 
			}

		case INTEGER: 
			switch(srcT){
				case DOUBLE: 
					return rdfcastToIntFromDouble(srcPtr, dstPtr); 
				case STRING: 
					return rdfcastToIntFromString(srcPtr, dstPtr); 	
				default: 
					return 0;
			}
		default:
			return 0; 
			
	}	
}

/*
 * Convert struct tm to time_t, keeping the time_zone information
 */

static
time_t tm2time(const struct tm *src)
{
	struct tm tmp;

	tmp = *src;

	return timegm(&tmp) - src->tm_gmtoff;
}

/*
 * Using strptime
 * */

int convertDateTimeToTimeT(char *sDateTime, int len, time_t *t){
	
	/*
	char* testDate[] = {
		"2001-10-26",
		"2001-10-26Z",
		"0001-10-26-02:00",
		"2001-10-26+02:00",
		"-2001-10-26",
		//"-20000-04-01",  	//Even this is valid xsd:date, we do not handle now
		"2001-10-26T21:32:52",
		"2001-10-26T21:32:52+02:00",
		"2001-10-26T19:32:52Z",
		"2001-10-26T19:32:52+00:00",
		"-2001-10-26T21:32:52"
		//"2001-10-26T21:32:52.12679"	//Even this is valid xsd:dateTime, we do not handle now
	};
	*/
	
	struct tm tm1 = {0};  

	char* acceptFormat[] = {
		"%F",		
		"-%F",
		"%FZ",
		"%F%z",
		"%FT%T",
		"%FT%TZ",
		"-%FT%T",
		"%FT%T%z",
	};

	int numAcceptFormat = 8;
	int j = 0; 	

	if (len > 21) j = 7;
	else if (len > 19) j = 5;
	else if (len > 12) j = 3;
	else if (len > 10) j = 1;
	
	for (; j < numAcceptFormat; j++){
		if ((strptime(sDateTime, acceptFormat[j], &tm1)) != NULL){		
			*t = tm2time(&tm1);
			return 1; 
		}
	}
	
	if (j == numAcceptFormat) printf("The %s is not a valid datetime string\n", sDateTime);
	
	return 0; 
}


void convertTimestampToLong(timestamp *ts, lng *t){

	int positiveDate = 0;
	int sign = 0;
	lng encodeLng = 0; 

	//Encoding timestamp to lng. 
	//First 4 bits are not used, 5th bits for sign of number of days value 
	//(1, if the number of days is negative)
	//27 bits for days, 32 bits for msecs
	if (ts->days < 0){
		positiveDate = 0 - ts->days;
		sign = 1; 
	} else {
		positiveDate = ts->days;
		sign = 0;
	}
	
	encodeLng |= (lng) positiveDate;
	encodeLng = encodeLng << (sizeof(ts->msecs) * 8); //Move 32 bits
	encodeLng |= (lng)sign << (sizeof(lng) * 8 - 5);	//Set the sign bit
	encodeLng = encodeLng | (lng) ts->msecs;	//Set 32 bits for msecs
	
	*t = encodeLng; 


}
/*
 * Using/extending monetdb mtime functions
 * */

int convertDateTimeToLong(char *sDateTime, lng *t){
	timestamp *ts = NULL; 
	//tzone *tz;
	int len, pos = 0; 

	
	char *p = NULL; 

	//printf("Checking datetime %s \n", sDateTime);
	
	//Remove the ending 'Z' from the string
	p = strchr(sDateTime, 'Z');
	
	len = (int) strlen(sDateTime);
	if (p != NULL) {
		if (p == (sDateTime + len - 1))
			*p = '\0';
	}

	/* Old way of computation 
	MTIMEtimestamp_fromstr(&ts, (const char* const*) &sDateTime);
	
	if (ts_isnil(ts) && len != 3){
		printf("The %s is not a valid datetime string\n", sDateTime);	
		return 0; 
	}
	*/
	
	//Same way as ql/backends/monet5/sql.c
	
	ts = (timestamp *) GDKmalloc(sizeof(timestamp)); 

	if (!sDateTime || strcmp(str_nil, sDateTime) == 0) {
		ts = timestamp_nil;
		return 0;
	}
	pos = timestamp_fromstr((const char*) sDateTime, &len, &ts);
	if (!pos){
		printf("The %s is not a valid datetime string (pos = %d)\n", sDateTime, pos);	
		if (ts) GDKfree(ts);
		return 0; 
	}
	if (ts_isnil(*ts) && len != 3){		//not a nil
		printf("The %s is not a valid datetime string (pos = %d)\n", sDateTime, pos);
		if (ts) GDKfree(ts);	
		return 0;
	}

	convertTimestampToLong(ts, t); 

	//printf("Encode string %s with days %d and msecs %d to lng %ld \n",sDateTime, ts->days, ts->msecs, *t);
	if (ts) GDKfree(ts); 
	
	return 1;
}

//TODO: For Datetime, we may use function in mtime in order to directly convert datetime string
//to mtime timestamp. Then, encode the timestamp in the oid, in which, 1 bit for sign of date, 27 bits for days value, 
//32 bits for msecs. (We can also use 28bits for msecs, and 32 bits for date). 
//However, we need to verify that mtime accepts rdf datetime format

void 
encodeValueInOid(ValPtr vrPtrRealValue, ObjectType objType, BUN* bun){

	int positiveInt = 0; 
	lng positiveLng = 0; 

	*bun = 0; 

	/* Add the type here by changing 3 bits at position 60, 61, 62 of oid */
	*bun |= (BUN)objType << (sizeof(BUN)*8 - 4);
	
	switch(objType){
		case INTEGER: 
			if (vrPtrRealValue->val.ival < 0){
				positiveInt = 0 - vrPtrRealValue->val.ival;				
				*bun |= (BUN) 1 << (sizeof(BUN)*8 - 5);	
			}
			else 
				positiveInt = vrPtrRealValue->val.ival;
			

			*bun |= (BUN) positiveInt;
			break;

		case DOUBLE:
			{
			double tmp = vrPtrRealValue->val.dval;
			BUN *tmpoid = (BUN *) & tmp;
			*tmpoid = *tmpoid >> 4;
			*bun |= *tmpoid; 
			}
			break;
		case DATETIME: 
			//Consider it is as lng value
			//No sign bit needed for this encoded lng
			assert (vrPtrRealValue->val.lval >= 0);
			positiveLng = vrPtrRealValue->val.lval;

			*bun |= (BUN) positiveLng;

			break;
		default:
			return; 

	}
}

static ObjectType getObjType_fromValRec(ValRecord v){
	ObjectType objT; 
	switch (v.vtype){
		case TYPE_bte:
		case TYPE_sht:
		case TYPE_int:
		case TYPE_wrd:
			objT = INTEGER; 
			break; 
		case TYPE_oid: 
			objT = URI; 
			break;
		case TYPE_lng:
		case TYPE_dbl:
		case TYPE_flt:			
			objT = DOUBLE;
			break; 
		case TYPE_str:		//Have not handle this case
			assert(0); 
		default: 
			assert(0); 
	}

	return objT; 
}
//Set the value for the new type from the old value
static void set_Val_of_new_type(ValPtr v, ObjectType objT){
	
	if (objT == INTEGER){
		switch (v->vtype){
			case TYPE_bte:
				v->val.ival = (int) v->val.btval;
				break;
			case TYPE_sht:
				v->val.ival = (int) v->val.shval;
				break;
			case TYPE_int:				
				break; 
			default: 
				assert(0); 
		}
		v->vtype = TYPE_int; 
	} else if (objT == DOUBLE) {
	
		switch (v->vtype){
			case TYPE_lng:
				v->val.dval = (double) v->val.lval;
				break;		
			case TYPE_flt:			
				v->val.dval = (double) v->val.fval;			
				break; 
			case TYPE_dbl:
				break;
			default: 
				assert(0);
		}
		v->vtype = TYPE_dbl; 				
	} else 
		assert(0);

}

void get_encodedOid_from_atom(atom *at, oid *ret){
	ValRecord vrec = at->data; 
	ObjectType objT = URI; 
	oid Min_String_Oid = RDF_MIN_LITERAL;
	Min_String_Oid |= (BUN)STRING << (sizeof(BUN)*8 - 4);

	objT = getObjType_fromValRec(vrec);

	if (objT == URI && (oid)(vrec.val.lval) >= Min_String_Oid){
		*ret = (oid)(vrec.val.lval); 
		return; 
	}
	
	if (objT == INTEGER || objT == DOUBLE){
		set_Val_of_new_type(&vrec, objT); 
		encodeValueInOid(&vrec, objT, ret); 
		return;
	}

}

void 
decodeValueFromOid(BUN bun, ObjectType objType, ValPtr vrPtrRealValue){
	BUN realval = 0; 
	int sign = 0; 
	int ival = 0; 
	lng lval = 0;
	double *realdbl = NULL; 
	
	//printf("Decode value from oid: "BUNFMT "\n",bun);
	switch(objType){
		case INTEGER: 
			sign = (int)((bun >> (sizeof(BUN)*8 - 5)) & 0x01);	//Get the sign bit
			//printf("Sign bit from "BUNFMT " is %d \n",bun, sign);
			realval = bun & (~((BUN) 0x1F << (sizeof(BUN)*8 - 5)));	//Get the real value 
			if (sign == 1) ival = 0 - realval ; 
			else	ival = realval; 
			VALset(vrPtrRealValue,TYPE_int, &ival);
			break;	
		case DOUBLE:
			realval = bun << 4; 
			realdbl = (double *) &realval;
			VALset(vrPtrRealValue,TYPE_dbl, realdbl);
			break;
		case DATETIME: 
			lval = bun & (~((BUN) 0x0F << (sizeof(BUN)*8 - 4))); //Get the real value by setting all first 4 bits to 0s
			VALset(vrPtrRealValue,TYPE_lng, &lval);
			break; 
		default:
			printf("The objecttype %d cannot be handled\n",objType);
			break; 
	
	}
}


void 
getStringFormatValueFromOid(BUN bun, ObjectType objType, str *obj){
        ValRecord       vrRealObjValue;
        ValRecord       vrCastedObjValue;

	decodeValueFromOid(bun, objType, &vrRealObjValue);

	if (rdfcast(objType, STRING, &vrRealObjValue, &vrCastedObjValue) != 1){
		printf("Every values should be able to be casted to String \n");
	}

	*obj = GDKstrdup(vrCastedObjValue.val.sval);
	
	VALclear(&vrRealObjValue);
	VALclear(&vrCastedObjValue);
}

	
/*
 * Convert value from tm format to timestamp of monet mtime
 * Only convert when storing in monetdb BAT for datetime
 * */
void convertTMtimeToMTime(time_t t, timestamp *ts){
	struct tm *timeinfo;
	char buf[128], *s1 = buf;
	*s1 = 0;

	timeinfo = gmtime(&t);
	//printf ( "Current local time and date: %s", asctime (timeinfo) );

	//Reformat the datetime string
	sprintf(s1, "%d-%02d-%02dT%d:%d:%d", timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday,
			timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);

	MTIMEtimestamp_fromstr(ts, (const char* const*) &s1);
}


/*
 * Convert value from tm format to timestamp of monet mtime
 * Only convert when storing in monetdb BAT for datetime
 * */
void convert_encodedLng_toTimestamp(lng t, timestamp *ts){
	int sign = 0; 
	int daypart = 0; 
	int msecpart = 0;

	sign = (int)((t >> (sizeof(BUN)*8 - 5)) & 0x01);      //Get the sign bit
	daypart = (int) (t >> (sizeof(int) * 8) );		      //Get 32 msb from t
	daypart = daypart & 0x07FFFFFF;			      //Clear first 5 bits to get real posi. value
	if (sign == 1) daypart = 0 - daypart; 
	msecpart = (int) (t & 0x00000000FFFFFFFF);

	ts->days = daypart;
	ts->msecs = msecpart; 
}

/*
static
char isIntWithQuote(char *input, int len){	// "123123" ==> INT
	
	int 	i;
	if (len < 3) return 0;
	for(i = 1; i < len-1; i++)
	{
		if(isdigit(input[i]) == 0){ // May also check ispunct(string[i]) != 0 
			//printf("NOT A DIGIT \n");
			break;
		}
	}
	//printf("i is %d \n",i);
	if(i == len)
		return 1;
	else
		return 0;
}
*/

