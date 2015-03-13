#include "monetdb_config.h"

#include "miniseed.h"
/* #include "vault.h" */
#include "mtime.h"

str MiniseedMount(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	bat** ret;

	int file_id = *(int*) getArgReference(stk,pci,4); /* arg 1: file_id of the input file. */
	str *targetfile = (str*) getArgReference(stk,pci,5); /* arg 2: string containing the input file path. */
	BAT *btime, *bdata, *bfile, *bseqno; /* BATs to return, representing columns of a table. */

	lng num_rows = 0;

	MSRecord *msr = NULL;
	MSFileParam *msfp = NULL;
	int retcode;
	int verbose = 1;
	int r;

	int seq_no_fake = 1;
	
	ret = (bat**) GDKmalloc(pci->retc*sizeof(bat*));
	if(ret == NULL)
	{
		throw(MAL,"miniseed.mount",MAL_MALLOC_FAIL);
	}

	for(r = 0; r < pci->retc; r++)
	{
		ret[r] = (int*) getArgReference(stk,pci,r);
	}

	cntxt = cntxt; /* to escape 'unused' parameter error. */
	mb = mb; /* to escape 'unused' parameter error. */

	bfile = BATnew(TYPE_void, TYPE_int, 0, PERSISTENT); /* create empty BAT for ret0. */
	if ( bfile == NULL)
		throw(MAL,"miniseed.mount",MAL_MALLOC_FAIL);
	BATseqbase(bfile, 0);
	bseqno = BATnew(TYPE_void, TYPE_int, 0, PERSISTENT); /* create empty BAT for ret1. */
	if ( bseqno == NULL)
		throw(MAL,"miniseed.mount",MAL_MALLOC_FAIL);
	BATseqbase(bseqno, 0);

	btime = BATnew(TYPE_void, TYPE_timestamp, 0, PERSISTENT); /* create empty BAT for ret2. */
	if ( btime == NULL)
		throw(MAL,"miniseed.mount",MAL_MALLOC_FAIL);
	BATseqbase(btime, 0);
	bdata = BATnew(TYPE_void, TYPE_int, 0, PERSISTENT); /* create empty BAT for ret3. */
	if ( bdata == NULL)
		throw(MAL,"miniseed.mount",MAL_MALLOC_FAIL);
	BATseqbase(bdata, 0);

	if(bfile == NULL || bseqno == NULL || btime == NULL || bdata == NULL) /* exception handling. */
	{
		if(bfile)
			BBPdecref(bfile->batCacheid, TRUE);
		if(bseqno)
			BBPdecref(bseqno->batCacheid, TRUE);
		if(btime)
			BBPdecref(btime->batCacheid, TRUE);
		if(bdata)
			BBPdecref(bdata->batCacheid, TRUE);
		throw(MAL,"miniseed.mount", MAL_MALLOC_FAIL);
	}

	/* loop through all records in the target mseed file. */
	while ((retcode = ms_readmsr_r (&msfp, &msr, *targetfile, 0, NULL, NULL, 1, 1, verbose)) == MS_NOERROR)
	{
		int seq_no = seq_no_fake;
		double sample_interval = HPTMODULUS / msr->samprate; /* calculate sampling interval from frequency */
		lng sampling_time = msr->starttime;

		lng num_samples = msr->numsamples;
		int *data_samples = msr->datasamples;

		lng i = 0;
		for(;i<num_samples;i++)
		{
			timestamp sampling_timestamp;
			lng st = sampling_time / 1000;
			MTIMEtimestamp_lng(&sampling_timestamp, &st);

			/* For each sample add one row to the table */
			BUNappend(bfile, (ptr) &file_id, FALSE);
			BUNappend(bseqno, (ptr) &seq_no, FALSE);
			BUNappend(btime, (ptr) &sampling_timestamp, FALSE);
			BUNappend(bdata, (ptr) (data_samples+i), FALSE);
			sampling_time += sample_interval;
		}

		num_rows += i;
		seq_no_fake++;

	}

	if ( retcode != MS_ENDOFFILE )
		ms_log (2, "Cannot read %s: %s\n", *targetfile, ms_errorstr(retcode));

	/* cleanup memory and close file */
	ms_readmsr_r (&msfp, &msr, NULL, 0, NULL, NULL, 0, 0, 0);

	printf("num_rows: "LLFMT"\n", num_rows);

	BBPkeepref(*ret[0] = bfile->batCacheid); /* return BAT. */
	BBPkeepref(*ret[1] = bseqno->batCacheid); /* return BAT. */
	BBPkeepref(*ret[2] = btime->batCacheid); /* return BAT. */
	BBPkeepref(*ret[3] = bdata->batCacheid); /* return BAT. */

	return MAL_SUCCEED;
}

str MiniseedMountSegmentMode(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	bat** ret;
	
	int file_id = *(int*) getArgReference(stk,pci,4); /* arg 1: file_id of the input file. */
	str *targetfile = (str*) getArgReference(stk,pci,5); /* arg 2: string containing the input file path. */
	BAT *btime, *bvalue, *bfile, *bsegid; /* BATs to return, representing columns of a table. */
	
	lng num_rows = 0;
	lng num_records = 0;
	
	MSTraceList *mstl = 0;
	MSRecord *msr = NULL;
	MSFileParam *msfp = NULL;
	int retcode;
	int verbose = 1;
	int r;
	int dataquality = 0;
	
	MSTraceID *id = 0;
	MSTraceSeg *seg = 0;
	int tracecnt = 0;
	
	int segment_id_fake = 0;
	
	short int data_flag = 1;
	
	double  timetol      = -1.0; /* Time tolerance for continuous traces */
	double  sampratetol  = -1.0; /* Sample rate tolerance for continuous traces */
	
	ret = (bat**) GDKmalloc(pci->retc*sizeof(bat*));
	if(ret == NULL)
	{
		throw(MAL,"miniseed.mount",MAL_MALLOC_FAIL);
	}
	
	for(r = 0; r < pci->retc; r++)
	{
		ret[r] = (int*) getArgReference(stk,pci,r);
	}
	
	cntxt = cntxt; /* to escape 'unused' parameter error. */
	mb = mb; /* to escape 'unused' parameter error. */
	
	bfile = BATnew(TYPE_void, TYPE_int, 0, PERSISTENT); /* create empty BAT for ret0. */
	if ( bfile == NULL)
		throw(MAL,"miniseed.mount",MAL_MALLOC_FAIL);
	BATseqbase(bfile, 0);
	bsegid = BATnew(TYPE_void, TYPE_int, 0, PERSISTENT); /* create empty BAT for ret1. */
	if ( bsegid == NULL)
		throw(MAL,"miniseed.mount",MAL_MALLOC_FAIL);
	BATseqbase(bsegid, 0);
	
	btime = BATnew(TYPE_void, TYPE_timestamp, 0, PERSISTENT); /* create empty BAT for ret2. */
	if ( btime == NULL)
		throw(MAL,"miniseed.mount",MAL_MALLOC_FAIL);
	BATseqbase(btime, 0);
	bvalue = BATnew(TYPE_void, TYPE_int, 0, PERSISTENT); /* create empty BAT for ret3. */
	if ( bvalue == NULL)
		throw(MAL,"miniseed.mount",MAL_MALLOC_FAIL);
	BATseqbase(bvalue, 0);
	
	if(bfile == NULL || bsegid == NULL || btime == NULL || bvalue == NULL) /* exception handling. */
	{
		if(bfile)
			BBPdecref(bfile->batCacheid, TRUE);
		if(bsegid)
			BBPdecref(bsegid->batCacheid, TRUE);
		if(btime)
			BBPdecref(btime->batCacheid, TRUE);
		if(bvalue)
			BBPdecref(bvalue->batCacheid, TRUE);
		throw(MAL,"miniseed.mount", MAL_MALLOC_FAIL);
	}
	
	mstl = mstl_init (NULL);
	
	/* loop through all records in the target mseed file. */
	while ((retcode = ms_readmsr_r (&msfp, &msr, *targetfile, 0, NULL, NULL, 1, data_flag, verbose)) == MS_NOERROR)
	{
		/* collect all records in a tracelist */
		mstl_addmsr (mstl, msr, dataquality, 1, timetol, sampratetol);
		
		num_records++;
	}
	
	if ( retcode != MS_ENDOFFILE )
		ms_log (2, "Cannot read %s: %s\n", *targetfile, ms_errorstr(retcode));
	
	if (!mstl)
	{
		ms_log (2, "miniseed.mount: mstl missing. File:%s, error:%s\n", *targetfile, ms_errorstr(retcode));
	}
	
	/* Loop through trace list */
	id = mstl->traces;
	while (id)
	{
		/* Loop through segment list */
		seg = id->first;
		while (seg)
		{
			/* mount */
			{
				int segment_id = segment_id_fake;
				double sample_interval = HPTMODULUS / seg->samprate; /* calculate sampling interval from frequency */
				lng sampling_time = seg->starttime;
				
				lng num_samples = seg->numsamples;
				int *data_samples = seg->datasamples;
				
				lng i = 0;
				for(;i<num_samples;i++)
				{
					timestamp sampling_timestamp;
					lng st = sampling_time / 1000;
					MTIMEtimestamp_lng(&sampling_timestamp, &st);
					
					/* For each sample add one row to the table */
					BUNappend(bfile, (ptr) &file_id, FALSE);
					BUNappend(bsegid, (ptr) &segment_id, FALSE);
					BUNappend(btime, (ptr) &sampling_timestamp, FALSE);
					BUNappend(bvalue, (ptr) (data_samples+i), FALSE);
					sampling_time += sample_interval;
				}
				
				num_rows += i;
			}
			
			segment_id_fake++;
			seg = seg->next;
		}
		tracecnt++;
		id = id->next;
	}
	
	if(tracecnt != 1)
		printf("Warning! Unexpectedly, trace count is not 1 in a file! Total %d traces in file %s", tracecnt, *targetfile);
	
	/* cleanup memory and close file */
	ms_readmsr_r (&msfp, &msr, NULL, 0, NULL, NULL, 0, 0, 0);
	if ( mstl )
		mstl_free (&mstl, 0);
	
	printf("num_segments: %d, num_records: "LLFMT", num_data_points: "LLFMT"\n", segment_id_fake, num_records, num_rows);
	
	BBPkeepref(*ret[0] = bfile->batCacheid); /* return BAT. */
	BBPkeepref(*ret[1] = bsegid->batCacheid); /* return BAT. */
	BBPkeepref(*ret[2] = btime->batCacheid); /* return BAT. */
	BBPkeepref(*ret[3] = bvalue->batCacheid); /* return BAT. */
	
	return MAL_SUCCEED;
}