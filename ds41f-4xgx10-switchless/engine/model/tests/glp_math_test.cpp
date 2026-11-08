// Host-only instruction-order regression. No model, no GPU, no eval prompts.
// Against recorded Owen v1 generated CUDA and tilelang/mhc.py, not the later v3 hook.
#include "model/glp.h"
#include "model/glp_math.h"
#include <array>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <set>
using namespace model::detail;
static uint32_t rng = 11322026;
static uint32_t next() { return rng = rng * 1664525u + 1013904223u; }
static float widen(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f,&u,4); return f; }
static uint16_t bf(float f) { uint32_t u;std::memcpy(&u,&f,4);return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
static float value() { return widen(uint16_t(((next() >> 31) << 15) | (uint32_t(123 + next()%9) << 7) | (next()%128))); }
static float mix() { return float(next()%1000001) / 1000000.f; }
static float mul(float a,float b) { volatile float v=a*b;return v; }
static float add(float a,float b) { volatile float v=a+b;return v; }
static float fold_oracle(float x,const float* r,float p,const float* c,int n) {
 float v=mul(c[n],r[0]);v=std::fma(p,x,v);
 for(int m=1;m<4;++m)v=std::fma(c[4*m+n],r[m],v);return v;
}
static float fold_r3(float x,const float* r,float p,const float* c,int n) {
 float v=add(mul(c[n],r[0]),mul(c[4+n],r[1]));v=add(v,mul(c[8+n],r[2]));v=add(v,mul(c[12+n],r[3]));return add(mul(p,x),v);
}
// Independent layout/reduction transcription of recorded generated CUDA. The
// fixed-path version reads the same shared column helper the new device code uses.
static float dot(const float* h,const float* d,int mode) {
 int threads=mode==0?256:320,per=5120/threads;
 std::vector<float> a(threads,0.f);
 for(int t=0;t<threads;++t)for(int j=0;j<per;++j) {
  int c=mode==0?t+256*j:mode==1?(j/8*320+t)*8+j%8:glp_column(t,j);
  a[t]=mode==2?glp_fma(h[c],d[c],a[t]):std::fma(h[c],d[c],a[t]);
 }
 for(int off=16;off;off/=2) {auto old=a;for(int t=0;t<threads;++t)a[t]=add(old[t],old[t^off]);}
 float total=0;for(int w=0;w<threads/32;++w)total=add(total,a[32*w]);return total;
}
int main(int argc,char**argv) {
 if(argc!=2)return 2;
 const auto file=glp_read(argv[1],40,5120);
 std::set<int> cols;for(int t=0;t<320;++t)for(int j=0;j<16;++j)cols.insert(glp_column(t,j));
 assert(cols.size()==5120 && *cols.begin()==0 && *cols.rbegin()==5119);
 size_t fold_n=0,fold_bad=0,dot_bad=0,project_bad=0,fixed_bad=0;
 float first_r3=0,first_owen=0; std::vector<float> h(5120),oldh(5120),r(4),comb(16),post(4);
 for(int layer:file.layers) {
  const float* d=file.dir.data()+layer*5120;
  for(int stream=0;stream<4;++stream) {
   for(int i=0;i<5120;++i) {
    for(auto&v:r)v=value();for(auto&v:comb)v=mix();for(auto&v:post)v=mix();float x=value();
    float want=widen(bf(fold_oracle(x,r.data(),post[stream],comb.data(),stream)));
    float got=widen(bf(glp_fold(x,r.data(),post[stream],comb.data(),stream)));
    h[i]=want;oldh[i]=widen(bf(fold_r3(x,r.data(),post[stream],comb.data(),stream)));++fold_n;
    fixed_bad+=bf(got)!=bf(want);fold_bad+=bf(oldh[i])!=bf(want);
   }
   float orig=dot(h.data(),d,0),owen=dot(h.data(),d,1),fixed=dot(h.data(),d,2);
   dot_bad+=std::memcmp(&orig,&owen,4)!=0;fixed_bad+=std::memcmp(&fixed,&owen,4)!=0;
   if(!first_r3&&orig!=owen){first_r3=orig;first_owen=owen;}
   for(int i=0;i<5120;++i) {
    auto out=[&](float a){return bf(add(h[i],-mul(mul(.75f,a),d[i])));};
    project_bad+=out(orig)!=out(owen);fixed_bad+=out(fixed)!=out(owen);
    assert(out(0)==bf(h[i]));
   }
  }
 }
 printf("{\"synthetic_fold_values\":%zu,\"r3_vs_owen_bf16_fold_differences\":%zu,\"dot_rows\":156,\"r3_vs_owen_fp32_dot_differences\":%zu,\"r3_vs_owen_bf16_projection_differences_same_fold\":%zu,\"fixed_vs_owen_host_differences\":%zu,\"first_dot_r3\":%.9g,\"first_dot_owen\":%.9g,\"column_coverage\":5120,\"alpha0_finite_identity\":true,\"gpu_execution\":false}\n",fold_n,fold_bad,dot_bad,project_bad,fixed_bad,first_r3,first_owen);
 return fixed_bad||!fold_bad||!dot_bad||!project_bad;
}
