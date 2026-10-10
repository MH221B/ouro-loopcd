#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <vector>

struct gguf_context;

struct ouro_hparams {
    int32_t n_vocab     = 0;
    int32_t n_embd      = 0;
    int32_t n_ff        = 0;
    int32_t n_head      = 0;
    int32_t n_head_kv   = 0;
    int32_t n_embd_head = 0;
    int32_t n_layer     = 0;
    int32_t n_loops     = 0;
    int32_t n_ctx       = 0;
    float   eps         = 0.0f;
    float   rope_theta  = 0.0f;
};

struct ouro_layer {
    ggml_tensor * attn_norm      = nullptr;
    ggml_tensor * attn_q         = nullptr;
    ggml_tensor * attn_k         = nullptr;
    ggml_tensor * attn_v         = nullptr;
    ggml_tensor * attn_output    = nullptr;
    ggml_tensor * attn_post_norm = nullptr;
    ggml_tensor * ffn_norm       = nullptr;
    ggml_tensor * ffn_gate       = nullptr;
    ggml_tensor * ffn_up         = nullptr;
    ggml_tensor * ffn_down       = nullptr;
    ggml_tensor * ffn_post_norm  = nullptr;
};

struct ouro_model {
    ouro_hparams hp;
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;

    ggml_context * ctx_w = nullptr;      // owns weight tensor structs
    gguf_context * ctx_g = nullptr;      // owns GGUF metadata
    ggml_backend_buffer_t buf_w = nullptr;

    ggml_tensor * tok_embd    = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output      = nullptr;
    std::vector<ouro_layer> layers;
};

// Creates the CPU backend and scheduler. Call before ouro_model_load.
bool ouro_backend_init(ouro_model & m, std::string & err);

// Reads the GGUF, validates shapes, and uploads weights to the backend.
bool ouro_model_load(const std::string & path, ouro_model & m, std::string & err);

void ouro_model_free(ouro_model & m);
