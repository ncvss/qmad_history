namespace qmad_history {

// the memory layout is U[mu,x,y,z,t1,g,h,t2] and v[x,y,z,t1,s,h,t2]
// t is split in pairs of 2 sites numbered by t1, the 2 parts of the pair are numbered by t2

// address for U in Grid format
inline int uixg (int t1, int mu, int g, int gi, int vol){
    return mu*vol*18 + t1*36 + g*12 + gi*4;
}
// address for v in Grid format
inline int vixg (int t1, int g, int s){
    return t1*48 + s*12 + g*4;
}
// address for hop pointer (this has only the t1)
inline int hixd (int t1, int h, int d){
    return t1*8 + h*2 + d;
}
// // address for gamfd
// inline int gixd (int mu, int s){
//     return mu*8 + s*2;
// }
// // address for sigfd
// inline int sixd (int munu, int s){
//     return munu*8 + s*2;
// }

// address for field strength tensors (index order F[t1,block number,triangle index,t2])
// the upper triangle is flattened with the following indices:
//  0 | 1 | 2 | 3 | 4 | 5
//  1 | 6 | 7 | 8 | 9 |10
//  2 | 7 |11 |12 |13 |14
//  3 | 8 |12 |15 |16 |17
//  4 | 9 |13 |16 |18 |19
//  5 |10 |14 |17 |19 |20
// (the lower triangles are the same numbers, but conjugated)
inline int fixg (int t1, int sblock, int triix){
    return t1*168 + sblock*84 + triix*4;
}


}
