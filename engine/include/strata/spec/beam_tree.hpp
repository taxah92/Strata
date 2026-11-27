// include/strata/spec/beam_tree.hpp - PR 6.2b: W=2..4 Adaptive Beam Tree pruning + EcoSpec sizing + realized-gain cooldown.
// Clean-room MIT implementation inspired by inkling2/cuda/ink_dspark.cuh (commit 5462a79).
#pragma once

#include "strata/kernels/verify_kernels.hpp"
#include "strata/spec/draft_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::spec {

struct BeamTreePlan {
    int T = 1;
    bool is_tree = false;
    int32_t tokens[strata::kernels::kTreeMaxNodes] = {};
    int32_t parent[strata::kernels::kTreeMaxNodes] = {};
    int32_t depth[strata::kernels::kTreeMaxNodes] = {};
    int8_t sibling_rank[strata::kernels::kTreeMaxNodes] = {};  // 0 = main chain, >= 1 = side branch
    float node_prob[strata::kernels::kTreeMaxNodes] = {};      // calibrated conditional prob
    int main_chain_len = 1;
    int extra_nodes = 0;
    double expected_tokens = 1.0;
    double score = 0.0;
};

class TreeGainTracker {
public:
    TreeGainTracker() {
        const char* v = std::getenv("STRATA_MTP_TREE");
        enabled_ = (v == nullptr || v[0] == '\0' || v[0] != '0');
        const char* m = std::getenv("STRATA_MTP_TREE_MAX");
        if (m != nullptr && m[0] != '\0') max_nodes_ = std::clamp(std::atoi(m), 2, strata::kernels::kTreeMaxNodes);
        const char* st = std::getenv("STRATA_MTP_TREE_STATS");
        stats_on_ = (st != nullptr && st[0] != '\0' && st[0] != '0');
    }

    bool enabled() const { return enabled_; }
    void set_enabled(bool on) { enabled_ = on; }
    int max_nodes() const { return max_nodes_; }
    void set_max_nodes(int n) { max_nodes_ = std::clamp(n, 2, strata::kernels::kTreeMaxNodes); }
    bool stats_on() const { return stats_on_; }

    bool tree_allowed() const { return enabled_ && cooldown_ == 0; }
    void tick_linear() {
        if (cooldown_ > 0) --cooldown_;
    }

    /// Online calibration multiplier for side-branch (top2..top4) draft probabilities.
    double alt_cal() const {
        return std::clamp(alt_ok_ / std::max(alt_exp_, 1e-6), 0.35, 1.35);
    }

    /// Observe outcome of a tree verify window: `path[0..n_keep-1]` is the accepted root-to-node path.
    void observe_tree(const BeamTreePlan& plan, const int32_t* path, int n_keep, double cost_tree, double cost_chain) {
        ++tree_windows_;
        if (plan.extra_nodes <= 0 || n_keep < 1) return;

        // Check whether accepted path diverged onto a side branch
        bool side_hit = false;
        int div_depth = 0;
        for (int k = 1; k < n_keep; ++k) {
            const int u = path[k];
            if (u >= 0 && u < plan.T && plan.sibling_rank[u] > 0) {
                side_hit = true;
                div_depth = k;
                break;
            }
        }
        const int saved_tok = side_hit ? (n_keep - div_depth) : 0;
        if (side_hit) {
            ++side_hits_;
            total_saved_tokens_ += saved_tok;
        }
        total_extra_nodes_ += plan.extra_nodes;

        // Calibrate alt_cal whenever the main chain rejected at a node that had side-branch children in the tree
        const int stop_node = side_hit ? plan.parent[path[div_depth]] : path[n_keep - 1];
        if (stop_node >= 0 && stop_node < plan.T && plan.sibling_rank[stop_node] == 0) {
            // Did stop_node have any side-branch children in the tree?
            double sum_alt_p = 0.0;
            for (int u = 1; u < plan.T; ++u) {
                if (plan.parent[u] == stop_node && plan.sibling_rank[u] > 0)
                    sum_alt_p += (double) plan.node_prob[u];
            }
            if (sum_alt_p > 0.01) {
                // Only update calibration when the main-chain child was actually rejected (either side_hit or n_keep < main_chain_len)
                const bool main_rejected = side_hit || (n_keep < plan.main_chain_len);
                if (main_rejected) {
                    alt_ok_ = 0.96 * alt_ok_ + (side_hit ? 1.0 : 0.0);
                    alt_exp_ = 0.96 * alt_exp_ + std::min(0.95, sum_alt_p);
                }
            }
        }

        // Realized-gain cooldown: compare realized saved tokens against marginal time overhead of extra nodes
        const double extra_ms = std::max(0.1, cost_tree - cost_chain);
        const double base_ms = std::max(1.0, cost_chain);
        const double needed_tok = (extra_ms / base_ms) * std::max(1.0, (double) n_keep - (double) saved_tok);
        rg_saved_ = 0.92 * rg_saved_ + (double) saved_tok;
        rg_needed_ = 0.92 * rg_needed_ + needed_tok;
        ++rg_rounds_;
        if (rg_rounds_ >= 16 && rg_needed_ > 0.05 && rg_saved_ < 0.70 * rg_needed_) {
            cooldown_ = 24;
            ++cooldown_triggers_;
            rg_saved_ = 0.0;
            rg_needed_ = 0.0;
            rg_rounds_ = 0;
        }
    }

    void report(const char* prefix = "strata serve: mtp beam tree") const {
        if (!stats_on_) return;
        std::fprintf(stderr, "%s: %lld tree windows, %lld side-branch hits, %lld saved tokens (%.2f/hit, %.2f extra nodes/win), "
                             "alt_cal=%.2f, cooldowns=%lld\n",
                     prefix, (long long) tree_windows_, (long long) side_hits_, (long long) total_saved_tokens_,
                     side_hits_ > 0 ? (double) total_saved_tokens_ / (double) side_hits_ : 0.0,
                     tree_windows_ > 0 ? (double) total_extra_nodes_ / (double) tree_windows_ : 0.0,
                     alt_cal(), (long long) cooldown_triggers_);
    }

    int64_t tree_windows() const { return tree_windows_; }
    int64_t side_hits() const { return side_hits_; }
    int64_t saved_tokens() const { return total_saved_tokens_; }

private:
    bool enabled_ = true;
    bool stats_on_ = false;
    int max_nodes_ = 6;
    double alt_ok_ = 3.6;
    double alt_exp_ = 4.0;
    double rg_saved_ = 0.0;
    double rg_needed_ = 0.0;
    int rg_rounds_ = 0;
    int cooldown_ = 0;
    int64_t tree_windows_ = 0;
    int64_t side_hits_ = 0;
    int64_t total_saved_tokens_ = 0;
    int64_t total_extra_nodes_ = 0;
    int64_t cooldown_triggers_ = 0;
};

namespace detail {

struct PoolNode {
    int32_t tok = 0;
    int parent = -1;       // index in pool
    int depth = 0;
    int8_t sib_rank = 0;
    float cond_p = 1.0f;
    double reach_p = 1.0;
};

// Reorder selected subset `sel[0..m-1]` (pool indices, sel[0]=0) into DFS pre-order and test GDN stack <= 2.
inline bool build_dfs_plan(const std::vector<PoolNode>& pool, const uint8_t* chosen, BeamTreePlan& out) {
    out = BeamTreePlan{};
    int n_pool = (int) pool.size();
    int dfs_order[strata::kernels::kTreeMaxNodes];
    int dfs_n = 0;

    // Iterative DFS preserving sibling order (main chain first -> rank 0 before side branches!)
    int stk[strata::kernels::kTreeMaxNodes * 2];
    int sp = 0;
    stk[sp++] = 0;
    while (sp > 0) {
        const int u = stk[--sp];
        if (dfs_n >= strata::kernels::kTreeMaxNodes) return false;
        dfs_order[dfs_n++] = u;
        // Push children in reverse order so lowest-index (main chain) is popped first!
        int ch[strata::kernels::kTreeMaxNodes];
        int nc = 0;
        for (int v = u + 1; v < n_pool; ++v) {
            if (chosen[v] && pool[(size_t) v].parent == u) ch[nc++] = v;
        }
        for (int k = nc - 1; k >= 0; --k) stk[sp++] = ch[k];
    }

    int remap[64];
    for (int i = 0; i < n_pool && i < 64; ++i) remap[i] = -1;
    for (int i = 0; i < dfs_n; ++i) remap[dfs_order[i]] = i;

    out.T = dfs_n;
    out.expected_tokens = 0.0;
    out.main_chain_len = 0;
    out.is_tree = false;
    for (int i = 0; i < dfs_n; ++i) {
        const PoolNode& pn = pool[(size_t) dfs_order[i]];
        out.tokens[i] = pn.tok;
        out.parent[i] = (i == 0) ? -1 : remap[pn.parent];
        out.depth[i] = pn.depth;
        out.sibling_rank[i] = pn.sib_rank;
        out.node_prob[i] = pn.cond_p;
        out.expected_tokens += pn.reach_p;
        if (pn.sib_rank == 0) ++out.main_chain_len;
        if (i > 0 && out.parent[i] != i - 1) out.is_tree = true;
    }
    out.extra_nodes = out.T - out.main_chain_len;
    strata::kernels::TreeVerifyMeta meta;
    return strata::kernels::build_tree_verify_meta(out.parent, out.T, meta);
}

}  // namespace detail

/// Build Bayes-optimal connected subtree (<= max_tree_nodes) maximizing E[tokens] / policy.cost_ms(M)
/// using main-chain drafts, zero-cost top-4 sibling leaves, and optional alt-branch continuations.
inline void build_optimal_beam_tree(
    int32_t root_tok,
    const int32_t* main_toks, const float* main_probs, int n_main,
    const int32_t* top4_ids, const float* top4_probs,   // [n_main][4]
    const int32_t* alt_toks, const float* alt_probs, int n_alt,  // continuations from top4_ids[0*4 + 1]
    const DraftPolicy& policy, const TreeGainTracker& tracker,
    int max_tree_nodes, int cap_linear, BeamTreePlan& best) {
    using namespace detail;
    const int max_m = std::clamp(max_tree_nodes, 1, strata::kernels::kTreeMaxNodes);
    const int lin_cap = std::clamp(cap_linear, 1, max_m);

    std::vector<PoolNode> pool;
    pool.reserve(32);
    pool.push_back(PoolNode{root_tok, -1, 0, 0, 1.0f, 1.0});

    // 1. Main chain nodes (pool indices 1..n_main)
    int chain_idx[16] = {};
    chain_idx[0] = 0;
    double q = 1.0;
    for (int j = 0; j < n_main && j + 1 < lin_cap; ++j) {
        const double a = policy.draft_rate((double) main_probs[j]);
        q *= a;
        chain_idx[j + 1] = (int) pool.size();
        pool.push_back(PoolNode{main_toks[j], chain_idx[j], j + 1, 0, (float) a, q});
    }

    // 2. Side-branch candidates (only if tree is allowed)
    int alt_root_pool_idx = -1;
    if (tracker.tree_allowed() && top4_ids != nullptr && top4_probs != nullptr) {
        const double acal = tracker.alt_cal();
        for (int j = 0; j < n_main && j < lin_cap - 1; ++j) {
            const int par = chain_idx[j];
            const double p_par = pool[(size_t) par].reach_p;
            if (p_par < 0.15) break;
            for (int k = 1; k < 4; ++k) {
                const int32_t tid = top4_ids[j * 4 + k];
                const float rp = top4_probs[j * 4 + k];
                if (tid < 0 || rp < 0.06f) continue;
                // Deduplicate among siblings of `par`
                bool dup = false;
                for (const auto& existing : pool) {
                    if (existing.parent == par && existing.tok == tid) { dup = true; break; }
                }
                if (dup) continue;
                const double a = std::clamp(acal * policy.draft_rate((double) rp), 0.0, 0.99);
                const double reach = p_par * a;
                if (reach < 0.05) continue;
                const int idx = (int) pool.size();
                if (j == 0 && k == 1) alt_root_pool_idx = idx;
                pool.push_back(PoolNode{tid, par, j + 1, (int8_t) k, (float) a, reach});
            }
        }
        // 3. Alt-branch continuations from root's top2 candidate (if drafted)
        if (alt_root_pool_idx > 0 && alt_toks != nullptr && alt_probs != nullptr && n_alt > 0) {
            int par = alt_root_pool_idx;
            double qa = pool[(size_t) par].reach_p;
            for (int j = 0; j < n_alt; ++j) {
                if (alt_toks[j] < 0) break;
                const double a = policy.draft_rate((double) alt_probs[j]);
                qa *= a;
                if (qa < 0.05) break;
                const int idx = (int) pool.size();
                pool.push_back(PoolNode{alt_toks[j], par, j + 2, 1, (float) a, qa});
                par = idx;
            }
        }
    }

    // Evaluate linear baseline first (best pure chain)
    best = BeamTreePlan{};
    best.T = 1;
    best.tokens[0] = root_tok;
    best.parent[0] = -1;
    best.depth[0] = 0;
    best.main_chain_len = 1;
    best.expected_tokens = 1.0;
    best.score = 1.0 / std::max(0.1, policy.cost_ms(1));

    const int n_pool = (int) pool.size();
    uint8_t chosen[64] = {};
    chosen[0] = 1;
    // Pure linear chain sizes 2..lin_cap
    for (int m = 2; m <= std::min(lin_cap, n_main + 1); ++m) {
        chosen[chain_idx[m - 1]] = 1;
        BeamTreePlan cand;
        if (build_dfs_plan(pool, chosen, cand)) {
            cand.score = cand.expected_tokens / std::max(0.1, policy.cost_ms(cand.T));
            if (cand.score > best.score) best = cand;
        }
    }
    const double best_linear_score = best.score;

    // Now evaluate Bayes-optimal greedy connected subtrees up to max_m
    if (tracker.tree_allowed() && n_pool > n_main + 1) {
        std::memset(chosen, 0, sizeof(chosen));
        chosen[0] = 1;
        int cur_m = 1;
        while (cur_m < max_m) {
            // Pick highest-reach_p unchosen node whose parent IS chosen and keeps GDN stack <= 2
            int best_u = -1;
            double best_p = -1.0;
            BeamTreePlan best_step_plan;
            for (int u = 1; u < n_pool && u < 64; ++u) {
                if (chosen[u]) continue;
                if (!chosen[pool[(size_t) u].parent]) continue;
                if (pool[(size_t) u].reach_p <= best_p) continue;
                chosen[u] = 1;
                BeamTreePlan trial;
                const bool ok = build_dfs_plan(pool, chosen, trial);
                chosen[u] = 0;
                if (ok) {
                    best_u = u;
                    best_p = pool[(size_t) u].reach_p;
                    best_step_plan = trial;
                }
            }
            if (best_u < 0) break;
            chosen[best_u] = 1;
            ++cur_m;
            best_step_plan.score = best_step_plan.expected_tokens / std::max(0.1, policy.cost_ms(best_step_plan.T));
            // Require a small 1.5% edge over pure linear chain before switching to tree graph
            const double thresh = best_step_plan.is_tree ? (best_linear_score * 1.015) : best_linear_score;
            if (best_step_plan.score > best.score && best_step_plan.score >= thresh) {
                best = best_step_plan;
            }
        }
    }
}

}  // namespace strata::spec
