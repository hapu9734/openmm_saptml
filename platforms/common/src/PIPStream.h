#ifndef PIP_STREAM_TABLES_H_
#define PIP_STREAM_TABLES_H_
// Host-side builder of the stream tables used by the stream-mode PIP kernels (wtpip/wdpip).
// The polynomial P(x~) = sum_m c_m prod_{v in m} x~_v (degree <= 4, variables 0..D-1, D = dummy of value 1)
// is written as P = 1/2 y^T A y with y = (1, x~_i, x~_i x~_j) and A symmetric (each monomial split into two
// factors from y, the split of degree-3/4 monomials chosen to reuse few pair factors).  Each row of A is
// stored as groups of entries sharing the first factor x~_g of their column, z_row = sum_g x~_g sum_j A x~_j:
//   info bits 0-5 inner variable j, 6-11 group variable of the next group, 12-17 / 18-23 row factors p, q,
//   bit 24 end of group, bit 25 end of row.
// The entry list is cut into numSegments row-aligned segments of similar length: (start, end, first group, rows).
#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace PIPStream {

struct Tables {
    std::vector<unsigned int> info;
    std::vector<double> coef;
    std::vector<int> segStart, segEnd, segFirstGroup, segRows;
    int numRows, numPairFactors;
};

inline int yId(int p, int q, int D) {
    if (p > q)
        std::swap(p, q);
    return p*(D+1)+q;
}

inline Tables build(const std::vector<std::vector<int> >& monoVars, const std::vector<double>& monoCoef,
                    int D, int numSegments) {
    std::map<std::pair<int, int>, int> freq;
    auto key = [](int a, int b) { return std::make_pair(std::min(a, b), std::max(a, b)); };
    for (const std::vector<int>& v : monoVars)
        if (v.size() == 4) {
            const int o[3][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
            for (int k = 0; k < 3; k++) {
                freq[key(v[o[k][0]], v[o[k][1]])]++;
                freq[key(v[o[k][2]], v[o[k][3]])]++;
            }
        }
        else if (v.size() == 3) {
            freq[key(v[0], v[1])]++;
            freq[key(v[0], v[2])]++;
            freq[key(v[1], v[2])]++;
        }
    std::set<std::pair<int, int> > used;
    auto score = [&](std::pair<int, int> p) { return (used.count(p) ? 1000000L : 0L)+(long) freq[p]; };
    std::map<std::pair<int, int>, double> A;
    auto add = [&](int ya, int yb, double c) {
        if (ya == yb)
            A[std::make_pair(ya, ya)] += 2*c;
        else {
            A[std::make_pair(ya, yb)] += c;
            A[std::make_pair(yb, ya)] += c;
        }
    };
    for (size_t m = 0; m < monoVars.size(); m++) {
        std::vector<int> v = monoVars[m];
        std::sort(v.begin(), v.end());
        double c = monoCoef[m];
        if (c == 0.0)
            continue;
        int d = (int) v.size();
        if (d == 0)
            add(yId(D, D, D), yId(D, D, D), c);
        else if (d == 1)
            add(yId(v[0], D, D), yId(D, D, D), c);
        else if (d == 2)
            add(yId(v[0], D, D), yId(v[1], D, D), c);
        else if (d == 3) {
            const int o[3][3] = {{0, 1, 2}, {0, 2, 1}, {1, 2, 0}};
            int best = 0;
            for (int k = 1; k < 3; k++)
                if (score(key(v[o[k][0]], v[o[k][1]])) > score(key(v[o[best][0]], v[o[best][1]])))
                    best = k;
            used.insert(key(v[o[best][0]], v[o[best][1]]));
            add(yId(v[o[best][0]], v[o[best][1]], D), yId(v[o[best][2]], D, D), c);
        }
        else {
            const int o[3][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
            int best = 0;
            long bestScore = -1;
            for (int k = 0; k < 3; k++) {
                long s = score(key(v[o[k][0]], v[o[k][1]]))+score(key(v[o[k][2]], v[o[k][3]]));
                if (s > bestScore) {
                    bestScore = s;
                    best = k;
                }
            }
            used.insert(key(v[o[best][0]], v[o[best][1]]));
            used.insert(key(v[o[best][2]], v[o[best][3]]));
            add(yId(v[o[best][0]], v[o[best][1]], D), yId(v[o[best][2]], v[o[best][3]], D), c);
        }
    }
    std::map<int, std::vector<std::pair<int, double> > > rows;
    for (const auto& e : A)
        if (e.second != 0.0)
            rows[e.first.first].push_back(std::make_pair(e.first.second, e.second));
    auto groupOf = [&](int col) { int p = col/(D+1), q = col%(D+1); return (q == D) ? D : p; };
    auto innerOf = [&](int col) { int p = col/(D+1), q = col%(D+1); return (q == D) ? p : q; };
    Tables t;
    t.numRows = (int) rows.size();
    t.numPairFactors = 0;
    std::vector<int> rowIds, rowStart;
    for (auto& r : rows) {
        std::sort(r.second.begin(), r.second.end(), [&](const std::pair<int, double>& a, const std::pair<int, double>& b) {
            int ga = groupOf(a.first), gb = groupOf(b.first);
            return (ga != gb) ? ga < gb : innerOf(a.first) < innerOf(b.first);
        });
        rowIds.push_back(r.first);
        if (r.first%(D+1) != D)
            t.numPairFactors++;
    }
    for (size_t ri = 0; ri < rowIds.size(); ri++) {
        const std::vector<std::pair<int, double> >& e = rows[rowIds[ri]];
        rowStart.push_back((int) t.info.size());
        int p = rowIds[ri]/(D+1), q = rowIds[ri]%(D+1);
        for (size_t k = 0; k < e.size(); k++) {
            unsigned int info = (unsigned int) innerOf(e[k].first);
            bool endRow = (k+1 == e.size());
            bool endGroup = endRow || groupOf(e[k+1].first) != groupOf(e[k].first);
            int next = 0;
            if (!endRow)
                next = groupOf(e[k+1].first);
            else if (ri+1 < rowIds.size())
                next = groupOf(rows[rowIds[ri+1]][0].first);
            info |= ((unsigned int) next) << 6;
            if (endGroup)
                info |= 1u << 24;
            if (endRow)
                info |= (((unsigned int) p) << 12) | (((unsigned int) q) << 18) | (1u << 25);
            t.info.push_back(info);
            t.coef.push_back(e[k].second);
        }
    }
    rowStart.push_back((int) t.info.size());
    int total = (int) t.info.size();
    numSegments = std::max(1, std::min(numSegments, t.numRows));
    int r = 0;
    for (int s = 0; s < numSegments; s++) {
        int target = (int) ((long long) total*(s+1)/numSegments);
        int r0 = r;
        while (r < t.numRows && (rowStart[r+1] <= target || r == r0))
            r++;
        if (s == numSegments-1)
            r = t.numRows;
        t.segStart.push_back(rowStart[r0]);
        t.segEnd.push_back(rowStart[r]);
        t.segFirstGroup.push_back(rowStart[r0] < total ? groupOf(rows[rowIds[r0]][0].first) : 0);
        t.segRows.push_back(r-r0);
        if (r >= t.numRows && s < numSegments-1) {          // ran out of rows: fewer segments
            break;
        }
    }
    return t;
}

} // namespace PIPStream
#endif
