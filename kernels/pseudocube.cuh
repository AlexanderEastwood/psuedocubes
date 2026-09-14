// Explicit device limbs; products use CUDA's exact high-half multiply.
struct U { unsigned long long lo,hi; };
__device__ U mul(unsigned long long a,unsigned long long b){return U{a*b,__umul64hi(a,b)};}
__device__ U add(U a,U b){unsigned long long low=a.lo+b.lo;return U{low,a.hi+b.hi+(low<a.lo)};}
__device__ U sub(U a,U b){return U{a.lo-b.lo,a.hi-b.hi-(a.lo<b.lo)};}
__device__ bool less(U a,U b){return a.hi<b.hi||(a.hi==b.hi&&a.lo<b.lo);}
extern "C" __global__ void arithmetic(const unsigned long long* in,unsigned long long* out,unsigned long long count){
  unsigned long long i=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;if(i>=count)return;
  U left=mul(in[4*i],in[4*i+1]),right=mul(in[4*i+2],in[4*i+3]),difference=sub(left,right);
  out[3*i]=difference.lo;out[3*i+1]=difference.hi;out[3*i+2]=less(left,right);
}
