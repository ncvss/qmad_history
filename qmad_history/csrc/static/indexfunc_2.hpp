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
