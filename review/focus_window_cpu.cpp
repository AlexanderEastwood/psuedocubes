// Isolated CPU producer: sorted partial CRT sums, two monotone range pointers.
#include <cstdint>
#include <cstddef>
extern "C" int focus_window(const uint64_t* C, uint64_t nc, const uint64_t* D, uint64_t nd,
                            uint64_t M, uint64_t lo, uint64_t hi, uint64_t offset,
                            uint64_t* output, uint64_t cap, uint64_t* count) {
    if (lo>hi || hi>M || M >= (uint64_t(1)<<63)) return 2;
    uint64_t written=0;
    for (unsigned wrap=0; wrap<2; ++wrap) {
        uint64_t a=lo+(wrap?M:0), b=hi+(wrap?M:0), left=nc, right=nc;
        for (uint64_t j=0; j<nd; ++j) {
            uint64_t d=D[j];
            if (d>=b) break;
            uint64_t lower=a>d?a-d:0, upper=b-d;
            while (left && C[left-1]>=lower) --left;
            while (right && C[right-1]>=upper) --right;
            for (uint64_t i=left; i<right; ++i) {
                if (written>=cap) { *count=written+1; return 1; }
                output[written++]=C[i]+d-(wrap?M:0)+offset;
            }
        }
    }
    *count=written;
    return 0;
}
