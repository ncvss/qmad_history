// index functions for flattened space time
// in old (mtsg) and new (tmgs) layout

// index for U in new layout
inline int uixn (int t, int mu, int g, int gi){
    return t*36 + mu*9 + g*3 + gi;
}
// index for v in new layout
inline int vixn (int t, int g, int s){
    return t*12 + g*4 + s;
}


// index for hops
inline int hix (int t, int m, int d){
    return t*8 + m*2 + d;
}



// address for U in old layout for std complex
inline int uixo (int t, int mu, int g, int gi, int vol){
    return mu*vol*9 + t*9 + g*3 + gi;
}
// address for v in old layout for std complex
inline int vixo (int t, int g, int s){
    return t*12 + s*3 + g;
}

// address for F in layout t,munu,g,gi
inline int fix (int t, int munu, int g, int gi){
    return t*54 + munu*9 + g*3 + gi;
}

// address for field strength tensors (index order F[t,triangle index,block number])
// the upper triangle is flattened with the following indices:
//  0 | 1 | 2 | 3 | 4 | 5
//  1 | 6 | 7 | 8 | 9 |10
//  2 | 7 |11 |12 |13 |14
//  3 | 8 |12 |15 |16 |17
//  4 | 9 |13 |16 |18 |19
//  5 |10 |14 |17 |19 |20
// (the lower triangles are the same numbers, but conjugated)
inline int sfix (int t, int sblock, int triix){
    return t*42 + triix*2 + sblock;
}
