/* Post-processing for a thresholded GNN bitstring:
 *   (1) conflict repair  - repeatedly drop the selected node with the most selected
 *                          neighbours until the set is a valid independent set
 *   (2) maximalization   - single pass adding any unselected node with no selected
 *                          neighbour (sufficient: selections are never removed in
 *                          this phase, so a node passed over stays dominated)
 * Both phases are charitable to the GNN: the reference implementation does neither,
 * it only counts violations. See DECISIONS.md D5.
 *
 * Input : graph.bin (int32 n, int32 m, 2m int32) and bits.bin (int32 n, n int8)
 * Output: JSON on stdout. Timed region excludes file I/O.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+1e-9*ts.tv_nsec;}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <graph.bin> <bits.bin>\n", argv[0]); return 1; }
    int n, m;
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr,"no graph\n"); return 1; }
    if (fread(&n,4,1,f)!=1 || fread(&m,4,1,f)!=1) return 1;
    int *elist = malloc((size_t)2*m*sizeof(int));
    if (fread(elist,sizeof(int),(size_t)2*m,f)!=(size_t)2*m) return 1;
    fclose(f);

    int nb; f = fopen(argv[2],"rb");
    if (!f) { fprintf(stderr,"no bits\n"); return 1; }
    if (fread(&nb,4,1,f)!=1 || nb!=n) { fprintf(stderr,"bit count mismatch %d vs %d\n",nb,n); return 1; }
    signed char *sel = malloc(n);
    if (fread(sel,1,n,f)!=(size_t)n) return 1;
    fclose(f);

    long long raw = 0; for (int v=0; v<n; v++) if (sel[v]) raw++;

    double t0 = now();
    int *xadj = calloc(n+1,sizeof(int)), *adjncy = malloc((size_t)2*m*sizeof(int));
    for (int i=0;i<m;i++){ xadj[elist[2*i]+1]++; xadj[elist[2*i+1]+1]++; }
    for (int v=0;v<n;v++) xadj[v+1]+=xadj[v];
    int *fill=malloc((n+1)*sizeof(int)); memcpy(fill,xadj,(n+1)*sizeof(int));
    for (int i=0;i<m;i++){int u=elist[2*i],v=elist[2*i+1];adjncy[fill[u]++]=v;adjncy[fill[v]++]=u;}
    free(fill);

    int maxdeg=0; for(int v=0;v<n;v++){int dg=xadj[v+1]-xadj[v]; if(dg>maxdeg)maxdeg=dg;}

    /* conflict degree = number of SELECTED neighbours, for selected nodes */
    int *cd = calloc(n,sizeof(int));
    for (int v=0; v<n; v++) if (sel[v])
        for (int i=xadj[v]; i<xadj[v+1]; i++) if (sel[adjncy[i]]) cd[v]++;

    /* bucket queue over conflict degree, drop highest first */
    int *bh = malloc((maxdeg+2)*sizeof(int)), *bn = malloc(n*sizeof(int)), *bp = malloc(n*sizeof(int));
    for (int i=0;i<=maxdeg+1;i++) bh[i]=-1;
    for (int v=0;v<n;v++) bn[v]=bp[v]=-1;
    #define LINK(v) do{int d=cd[v];bp[v]=-1;bn[v]=bh[d];if(bh[d]>=0)bp[bh[d]]=v;bh[d]=v;}while(0)
    #define UNLINK(v) do{int d=cd[v];if(bp[v]>=0)bn[bp[v]]=bn[v];else bh[d]=bn[v];if(bn[v]>=0)bp[bn[v]]=bp[v];bp[v]=bn[v]=-1;}while(0)
    for (int v=0;v<n;v++) if (sel[v] && cd[v]>0) LINK(v);

    long long dropped = 0;
    int top = maxdeg;
    while (top >= 1) {
        if (bh[top] < 0) { top--; continue; }
        int v = bh[top];
        UNLINK(v);
        sel[v] = 0; dropped++;
        for (int i=xadj[v]; i<xadj[v+1]; i++) {
            int w = adjncy[i];
            if (sel[w] && cd[w] > 0) { UNLINK(w); cd[w]--; if (cd[w] > 0) LINK(w); }
        }
        if (top < maxdeg) top = maxdeg;   /* degrees only decrease, but restart is cheap */
    }

    /* maximalize */
    long long added = 0;
    for (int v=0; v<n; v++) {
        if (sel[v]) continue;
        int ok = 1;
        for (int i=xadj[v]; i<xadj[v+1]; i++) if (sel[adjncy[i]]) { ok=0; break; }
        if (ok) { sel[v]=1; added++; }
    }
    double t_post = now() - t0;

    long long final=0, viol=0, nonmax=0;
    for (int v=0;v<n;v++) if (sel[v]) final++;
    for (int i=0;i<m;i++) if (sel[elist[2*i]] && sel[elist[2*i+1]]) viol++;
    for (int v=0;v<n;v++){ if(sel[v])continue; int ok=1;
        for(int i=xadj[v];i<xadj[v+1];i++) if(sel[adjncy[i]]){ok=0;break;} if(ok)nonmax++; }

    printf("{\"n\":%d,\"raw_size\":%lld,\"repaired_size\":%lld,\"dropped\":%lld,\"added\":%lld,"
           "\"t_post_s\":%.6f,\"violations_after\":%lld,\"nonmaximal_after\":%lld,\"density\":%.10f}\n",
           n, raw, final, dropped, added, t_post, viol, nonmax, (double)final/n);
    return 0;
}
