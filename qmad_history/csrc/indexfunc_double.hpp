// address for complex gauge field pointer that is stored as 2 doubles
inline __attribute__((always_inline)) int uixd (int t, int mu, int g, int gi){
    return t*72 + mu*18 + g*6 + gi*2;
}
// address for complex vector field pointer that is stored as 2 doubles
inline __attribute__((always_inline)) int vixd (int t, int g, int s){
    return t*24 + g*8 + s*2;
}
// address for hop pointer
inline __attribute__((always_inline)) int hixd (int t, int h, int d){
    return t*8 + h*2 + d;
}
// address for gamfd
inline __attribute__((always_inline)) int gixd (int mu, int s){
    return mu*8 + s*2;
}



// address for U in old layout for 2-double format
inline int uixo (int t, int mu, int g, int gi, int vol){
    return mu*vol*18 + t*18 + g*6 + gi*2;
}
// address for v in old layout for 2-double format
inline int vixo (int t, int g, int s){
    return t*24 + s*6 + g*2;
}
