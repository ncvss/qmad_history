// index functions for flattened space time
// in old (mtsg) layout

namespace qmad_history{

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

}
