#include <stdint.h>
/* CPU __uint128_t reference for the explicit device limb implementation. */
int products(uint64_t a,uint64_t b,uint64_t c,uint64_t d,uint64_t *low,uint64_t *high){
    __uint128_t left=(__uint128_t)a*b,right=(__uint128_t)c*d;
    __uint128_t difference=left-right;*low=(uint64_t)difference;*high=(uint64_t)(difference>>64);
    return left<right;
}
