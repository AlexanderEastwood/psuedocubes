/* Independent byte-segment coprimality sieve. Arguments are inclusive ROOT bounds. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#define SEG 1048576ULL
static uint64_t parse(const char *s) {
    char *e; errno=0; unsigned long long x=strtoull(s,&e,10);
    if(errno || *e || *s=='-') {fprintf(stderr,"invalid integer\n");exit(2);} return x;
}
int main(int argc,char **argv) {
    if(argc<4 || argc>5)return 2;
    uint64_t first=parse(argv[1]),last=parse(argv[2]),p=parse(argv[3]);
    int emit=argc==5 && !strcmp(argv[4],"--list");
    if(first<1 || last<first || last>UINT32_MAX || p<2 || p>2000 || (emit && last-first>100000))return 2;
    unsigned char composite[2001]={0}; unsigned primes[400],np=0;
    for(unsigned q=2;q<=p;q++)if(!composite[q]){
        primes[np++]=q; for(unsigned m=q*q;m<=p;m+=q)composite[m]=1;
    }
    unsigned char *flags=malloc(SEG); if(!flags)return 3;
    uint64_t count=0; int comma=0;
    printf("{\"algorithm\":\"byte-segment\",\"root_first\":\"%llu\",\"root_last\":\"%llu\",\"p\":%llu,",(unsigned long long)first,(unsigned long long)last,(unsigned long long)p);
    if(emit)printf("\"roots\":[");
    for(uint64_t start=first;start<=last;){
        uint64_t n=last-start+1;if(n>SEG)n=SEG;memset(flags,1,n);
        for(unsigned i=0;i<np;i++){
            uint64_t q=primes[i],m=((start+q-1)/q)*q;
            for(;m<start+n;m+=q)flags[m-start]=0;
        }
        for(uint64_t i=0;i<n;i++)if(flags[i]){
            count++;if(emit){printf("%s\"%llu\"",comma?",":"",(unsigned long long)(start+i));comma=1;}
        }
        start+=n;
    }
    if(emit)printf("],");
    printf("\"count\":\"%llu\",\"working_bytes\":%llu}\n",(unsigned long long)count,(unsigned long long)SEG);free(flags);return 0;
}
