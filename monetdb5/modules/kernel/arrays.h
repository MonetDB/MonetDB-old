#ifndef _ARRAYS_H
#define _ARRAYS_H


#ifdef WIN32
#if !defined(LIBMAL) && !defined(LIBATOMS) && !defined(LIBKERNEL) && !defined(LIBMAL) && !defined(LIBOPTIMIZER) && !defined(LIBSCHEDULER) && !defined(LIBMONETDB5)
#define algebra_export extern __declspec(dllimport)
#else
#define algebra_export extern __declspec(dllexport)
#endif
#else
#define algebra_export extern
#endif

algebra_export str ALGmaterialise(bat* mbrResult, const ptr *dimsCands, const ptr *dims) ;

algebra_export str ALGdimensionLeftfetchjoin1(bat* result, const ptr *dimsCands, const ptr *dim, const ptr *dims) ;
algebra_export str ALGdimensionLeftfetchjoin2(bat* result, const bat *oidsCands, const ptr *dimsCands, const ptr *dim, const ptr *dims) ;
algebra_export str ALGdimensionLeftfetchjoin3(bat* result, const ptr* dimsCands, const ptr *array);

algebra_export str ALGnonDimensionLeftfetchjoin1(bat* result, const bat* mbrOids, const bat *vals, const ptr *dims);
algebra_export str ALGnonDimensionLeftfetchjoin2(bat* result, const ptr *array, const bat *vals, const ptr *dims);
//algebra_export str ALGnonDimensionLeftfetchjoin3(bat* result, const ptr *dimsCands, const bat *vals, const ptr *dims);

algebra_export str ALGdimensionSubselect2(ptr *dimsRes, const ptr *dim, const ptr* dims, const ptr *dimsCands,
                            const void *low, const void *high, const bit *li, const bit *hi, const bit *anti);
algebra_export str ALGdimensionSubselect1(ptr *dimsRes, const ptr *dim, const ptr* dims, 
                            const void *low, const void *high, const bit *li, const bit *hi, const bit *anti);
//algebra_export str ALGdimensionSubselect3(ptr *dimsRes, bat *oidsRes, const ptr *array, const bat *vals,
//							const void *low, const void *high, const bit *li, const bit *hi, const bit *anti);

algebra_export str ALGdimensionThetasubselect2(ptr *dimsRes, const ptr *dim, const ptr* dims, const ptr *dimsCand,
											const void *val, const char **op);
algebra_export str ALGdimensionThetasubselect1(ptr *dimsRes, const ptr *dim, const ptr* dims, 
											const void *val, const char **op);

algebra_export str ALGnonDimensionSubselect1(bat *oidsRes, const bat *values, const ptr *dims, 
							const void *low, const void *high, const bit *li, const bit *hi, const bit *anti);
algebra_export str ALGnonDimensionSubselect2(bat *oidsRes, const bat *values, const ptr *dims, const bat* oidsCands, 
							const void *low, const void *high, const bit *li, const bit *hi, const bit *anti);
algebra_export str ALGnonDimensionSubselect3(bat *oidsRes, ptr *dimsRes, const bat* values, const ptr *dims, const ptr *dimsCands, 
                            const void *low, const void *high, const bit *li, const bit *hi, const bit *anti);
algebra_export str ALGnonDimensionSubselect4(bat *oidsRes, ptr *dimsRes, const bat* values, const ptr *dims, const bat* oidsCands, const ptr *dimsCands, 
                            const void *low, const void *high, const bit *li, const bit *hi, const bit *anti);

algebra_export str ALGnonDimensionThetasubselect1(bat *oidsRes, const bat* vals, const ptr *dims, 
			                            const void *val, const char **op);
algebra_export str ALGnonDimensionThetasubselect2(bat *oidsRes, const bat* vals, const ptr *dims, const bat* oidsCands, 
			                            const void *val, const char **op);
algebra_export str ALGnonDimensionThetasubselect3(bat *oidsRes, ptr *dimsRes, const bat* vals, const ptr *dims, const ptr *dimsCands, 
			                            const void *val, const char **op);
algebra_export str ALGnonDimensionThetasubselect4(bat *oidsRes, ptr *dimsRes, const bat* vals, const ptr *dims, const bat* oidsCands, const ptr *dimsCands, 
			                            const void *val, const char **op);

algebra_export str ALGsubjoin1(ptr *dimsResL, ptr *dimsResR, const ptr *dimL, const ptr *dimsL, const ptr *dimR, const ptr *dimsR);
algebra_export str ALGsubjoin2(ptr *dimsResL, ptr *dimsResR, const ptr *dimsCandsL, const ptr *dimL, const ptr *dimsL, const ptr *dimsCandsR, const ptr *dimR, const ptr *dimsR);

algebra_export str ALGsubrangejoin1(ptr *dimsResL, ptr *dimsResR, const ptr *dimL, const ptr *dimsL, const ptr *dimR1, const ptr *dimR2, const ptr *dimsR, const bat *sl, const bat *sr, const bit *li, const bit *hi, const lng *estimate);
algebra_export str ALGsubrangejoin2(ptr *dimsResL, ptr *dimsResR, const ptr *dimL, const ptr *dimsL, const ptr *dimR1, const ptr *dimR2, const ptr *dimsR, const ptr *dimsCandL, const ptr *dimsCandR, const bit *li, const bit *hi, const lng *estimate);

algebra_export str ALGprojectDimension(bat* result, const ptr *dim, const ptr *array);
algebra_export str ALGprojectNonDimension(bat *result, const bat *vals, const ptr *array);


algebra_export str ALGnonDimensionQRDecomposition(bat *oidsRes, ptr *dimsRes, const bat* vals, const ptr *dims);

algebra_export str ALGmbr1(ptr *mbrRes, const bat *oidsCands, const ptr *dimsCands, const ptr *dims);
algebra_export str ALGmbr2(ptr *mbrRes, const bat *oidsCands, const ptr *dims);
algebra_export str ALGouterjoin(bat *res, const bat *l, const bat *r);

algebra_export str ALGarrayCount(wrd *res, const ptr *array);
//algebra_export str ALGproject(bat *result, const ptr* candDims, const bat* candBAT);
#endif /* _ARRAYS_H */
