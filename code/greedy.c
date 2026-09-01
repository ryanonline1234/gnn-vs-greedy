/* Degree-based greedy (DGA) and random greedy (GA) for Maximum Independent Set
 * on sparse graphs, with a bucket queue -> near-linear time.
 *
 * This is the baseline champion from Angelini & Ricci-Tersenghi, NMI 5, 29 (2023).
 * Implemented in C because the timing comparison is the crux of the dispute and a
 * slow baseline would bias the result toward the GNN.
 *
 * Input: binary file  int32 n, int32 m, then 2*m int32 (u,v) pairs.
 * Output: JSON on stdout.
 *
 * Timing: excludes file read; includes CSR construction + the greedy loop.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* xorshift128+ so the random-greedy tie-breaking is reproducible and fast */
static unsigned long long S0, S1;
static void seed_rng(unsigned long long s) {
    S0 = s * 6364136223846793005ULL + 1442695040888963407ULL;
    S1 = s ^ 0x9E3779B97F4A7C15ULL;
    if (!S0) S0 = 1; if (!S1) S1 = 2;
}
static unsigned long long rnd(void) {
    unsigned long long x = S0, y = S1;
    S0 = y; x ^= x << 23; S1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return S1 + y;
}

int    n, m, maxdeg;
int   *xadj, *adjncy;      /* CSR */
int   *deg;                /* current degree among alive nodes */
char  *alive;
/* bucket queue: doubly linked list per degree value */
int   *bhead, *bnext, *bprev;
int    nalive;
/* random-greedy needs uniform selection among alive: compact alive array */
int   *alive_list, *alive_pos;

static void bucket_unlink(int v) {
    int d = deg[v];
    if (bprev[v] >= 0) bnext[bprev[v]] = bnext[v]; else bhead[d] = bnext[v];
    if (bnext[v] >= 0) bprev[bnext[v]] = bprev[v];
    bprev[v] = bnext[v] = -1;
}
static void bucket_link(int v) {
    int d = deg[v];
    bprev[v] = -1; bnext[v] = bhead[d];
    if (bhead[d] >= 0) bprev[bhead[d]] = v;
    bhead[d] = v;
}

static void alive_remove(int v) {
    int p = alive_pos[v], last = alive_list[nalive - 1];
    alive_list[p] = last; alive_pos[last] = p;
    nalive--; alive_pos[v] = -1;
}

/* Delete node u from the residual graph, updating neighbour degrees. */
static void delete_node(int u) {
    alive[u] = 0;
    bucket_unlink(u);
    alive_remove(u);
    for (int i = xadj[u]; i < xadj[u + 1]; i++) {
        int w = adjncy[i];
        if (alive[w]) { bucket_unlink(w); deg[w]--; bucket_link(w); }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <graph.bin> <dga|ga> [seed] [--dump out.bin]\n", argv[0]); return 1; }
    const char *mode = argv[2];
    unsigned long long seed = (argc > 3) ? strtoull(argv[3], NULL, 10) : 0;
    const char *dump = NULL;
    for (int i = 3; i < argc - 1; i++) if (!strcmp(argv[i], "--dump")) dump = argv[i + 1];

    /* ---- read graph (NOT timed) ---- */
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    if (fread(&n, 4, 1, f) != 1 || fread(&m, 4, 1, f) != 1) return 1;
    int *elist = malloc((size_t)2 * m * sizeof(int));
    if (fread(elist, sizeof(int), (size_t)2 * m, f) != (size_t)2 * m) return 1;
    fclose(f);

    /* ---- timed region begins: CSR build + greedy ---- */
    double t0 = now();
    xadj   = calloc(n + 1, sizeof(int));
    adjncy = malloc((size_t)2 * m * sizeof(int));
    deg    = malloc(n * sizeof(int));
    alive  = malloc(n);
    bnext  = malloc(n * sizeof(int));
    bprev  = malloc(n * sizeof(int));
    alive_list = malloc(n * sizeof(int));
    alive_pos  = malloc(n * sizeof(int));

    for (int i = 0; i < m; i++) { xadj[elist[2*i]+1]++; xadj[elist[2*i+1]+1]++; }
    for (int v = 0; v < n; v++) xadj[v+1] += xadj[v];
    int *fill = malloc((n + 1) * sizeof(int));
    memcpy(fill, xadj, (n + 1) * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = elist[2*i], v = elist[2*i+1];
        adjncy[fill[u]++] = v; adjncy[fill[v]++] = u;
    }
    free(fill);

    maxdeg = 0;
    for (int v = 0; v < n; v++) {
        deg[v] = xadj[v+1] - xadj[v];
        if (deg[v] > maxdeg) maxdeg = deg[v];
        alive[v] = 1; alive_list[v] = v; alive_pos[v] = v;
    }
    nalive = n;
    bhead = malloc((maxdeg + 2) * sizeof(int));
    for (int i = 0; i <= maxdeg + 1; i++) bhead[i] = -1;
    for (int v = 0; v < n; v++) { bprev[v] = bnext[v] = -1; bucket_link(v); }

    seed_rng(seed);
    int *is_set = malloc(n * sizeof(int));
    int is_n = 0;
    char *chosen = calloc(n, 1);
    int  *nbrs = malloc((maxdeg + 1) * sizeof(int));

    int use_dga = !strcmp(mode, "dga");
    while (nalive > 0) {
        int v = -1;
        if (use_dga) {
            for (int d = 0; d <= maxdeg; d++) if (bhead[d] >= 0) { v = bhead[d]; break; }
        } else {
            v = alive_list[rnd() % (unsigned long long)nalive];
        }
        is_set[is_n++] = v; chosen[v] = 1;
        int k = 0;
        for (int i = xadj[v]; i < xadj[v+1]; i++) if (alive[adjncy[i]]) nbrs[k++] = adjncy[i];
        delete_node(v);
        for (int i = 0; i < k; i++) if (alive[nbrs[i]]) delete_node(nbrs[i]);
    }
    double t_solve = now() - t0;
    /* ---- timed region ends ---- */

    /* verification (not timed): independence + maximality */
    long long viol = 0;
    for (int i = 0; i < m; i++) if (chosen[elist[2*i]] && chosen[elist[2*i+1]]) viol++;
    long long nonmaximal = 0;
    for (int v = 0; v < n; v++) {
        if (chosen[v]) continue;
        int ok = 1;
        for (int i = xadj[v]; i < xadj[v+1]; i++) if (chosen[adjncy[i]]) { ok = 0; break; }
        if (ok) nonmaximal++;
    }

    if (dump) {
        FILE *g = fopen(dump, "wb");
        fwrite(&is_n, 4, 1, g); fwrite(is_set, 4, is_n, g); fclose(g);
    }
    printf("{\"method\":\"%s\",\"n\":%d,\"m\":%d,\"size\":%d,\"density\":%.10f,"
           "\"t_solve_s\":%.6f,\"violations\":%lld,\"nonmaximal\":%lld,\"seed\":%llu}\n",
           mode, n, m, is_n, (double)is_n / n, t_solve, viol, nonmaximal, seed);
    return 0;
}
