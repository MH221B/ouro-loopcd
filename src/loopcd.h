#pragma once

#include "model.h"

#include <string>
#include <vector>

enum ouro_loopcd_mode {
    OURO_LOOPCD_OFF    = 0,
    OURO_LOOPCD_LOGITS = 1,
    OURO_LOOPCD_HIDDEN = 2,
};

// LoopCD hyperparameters (spec 4.4; paper Appendix B.4).
struct ouro_loopcd_params {
    ouro_loopcd_mode mode = OURO_LOOPCD_LOGITS;
    float omega     = 0.3f;   // fixed strength
    float omega_max = 1.0f;   // adaptive cap (LoopCD-Logits only)
    bool  adaptive  = false;  // omega = omega_max * (1 - (p1 - p2)) from softmax(zR), Eq. (4)
    int   ref_iter  = 1;      // reference loop (1-based); only 1 is exposed by the M2 taps
};

// All arrays are ne0-fastest: element (k, i) lives at k + dim*i.

// LoopCD-Logits, Eq. (3): out = zR + omega*(zR - z1). dim = n_vocab.
void ouro_loopcd_combine_logits(const float * zR, const float * z1,
                                int n_vocab, int n, float omega, float * out);

// LoopCD-Hidden, Eq. (2) on the PRE-norm loop outputs: out = gR + omega*(gR - g1). dim = n_embd.
// The output norm is NOT applied here; the caller feeds `out` to ouro_loopcd_hidden_head, which
// applies output_norm and then the head. (Combining the post-norm states and then the linear head
// would instead reproduce LoopCD-Logits exactly.)
void ouro_loopcd_combine_hidden(const float * gR, const float * g1,
                                int n_embd, int n, float omega, float * out);

// Adaptive strength, Eq. (4): omega = omega_max*(1 - (p1 - p2)) from softmax over zR_col (n_vocab).
float ouro_loopcd_adaptive_omega(const float * zR_col, int n_vocab, float omega_max);

// Hidden-form head: z' = m.output * rms_norm(g', m.output_norm, eps) on the device. g' is the
// combined PRE-norm state, [n_embd, n]; z_out becomes [n_vocab, n]. This is the paper's
// lm_head = final_norm . vocab_projection with Ouro's empty coda. Reuses m.output, m.output_norm,
// m.hp.eps and m.sched. Taps read from the M2 forward are invalidated by this call.
bool ouro_loopcd_hidden_head(ouro_model & m, const float * g, int n,
                             std::vector<float> & z_out, std::string & err);
