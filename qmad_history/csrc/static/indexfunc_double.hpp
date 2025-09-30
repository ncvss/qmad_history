// functions to compute the flattened address from the indices
// for arrays where complex numbers are stored as 2 adjacent doubles
// in old (mtsg) and new (tmgs) layout

// address for complex gauge field pointer that is stored as 2 doubles
// in tmgh layout
inline int uixd (int t, int mu, int g, int gi){
    return t*72 + mu*18 + g*6 + gi*2;
}
// address for complex vector field pointer that is stored as 2 doubles
// in tgs layout
inline int vixd (int t, int g, int s){
    return t*24 + g*8 + s*2;
}
// address for hop pointer
inline int hixd (int t, int h, int d){
    return t*8 + h*2 + d;
}
// address for gamfd
inline int gixd (int mu, int s){
    return mu*8 + s*2;
}
// address for sigfd
inline int sixd (int munu, int s){
    return munu*8 + s*2;
}
// address for field strength tensors (index order F[t,munu,g,gi])
inline int fixd (int t, int munu, int g, int gi){
    return t*108 + munu*18 + g*6 + gi*2;
}


// address for complex gauge field pointer that is stored as 2 doubles
// in mtgh layout
inline int uixo (int t, int mu, int g, int gi, int vol){
    return mu*vol*18 + t*18 + g*6 + gi*2;
}
// address for complex vector field pointer that is stored as 2 doubles
// in tsg layout
inline int vixo (int t, int g, int s){
    return t*24 + s*6 + g*2;
}
