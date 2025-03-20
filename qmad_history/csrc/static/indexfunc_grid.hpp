namespace qmad_history {

// the memory layout is U[mu,x,y,z,t1,g,h,t2] and v[x,y,z,t1,s,h,t2]
// t is split in pairs of 2 sites numbered by t1, the 2 parts of the pair are numbered by t2

// address for U in grid format
inline int uixg (int t1, int mu, int g, int gi, int vol){
    return mu*vol*18 + t1*36 + g*12 + gi*4;
}
// address for v in grid format
inline int vixg (int t1, int g, int s){
    return t1*48 + s*12 + g*4;
}
// address for hop pointer (this has only the t1)
inline int hixd (int t1, int h, int d){
    return t1*8 + h*2 + d;
}
// address for gamfd
inline int gixd (int mu, int s){
    return mu*8 + s*2;
}
// address for sigfd
inline int sixd (int munu, int s){
    return munu*8 + s*2;
}
// address for field strength tensors (index order F[t1,munu,g,gi,t2])
inline int fixg (int t1, int munu, int g, int gi){
    return t1*216 + munu*36 + g*12 + gi*4;
}



}
