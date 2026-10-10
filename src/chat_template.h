#pragma once

#include <string>
#include <vector>

struct ouro_chat_message {
    std::string role;      // "system" | "user" | "assistant"
    std::string content;
};

// Renders the Ouro template (tokenizer_config.json), matching HF `apply_chat_template`:
//   {%- if messages[0].role == 'system' -%}<|im_start|>system\n{content}<|im_end|>\n
//    {%- else -%}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{%- endif -%}
//   {%- for message -%}{%- if message.role=='system' and loop.first -%}{# skip #}
//    {%- else -%}<|im_start|>{role}\n{content}<|im_end|>\n{%- endif -%}{%- endfor -%}
//   {%- if add_generation_prompt -%}<|im_start|>assistant\n
//    {%- if enable_thinking -%}<think>\n{%- endif -%}{%- endif -%}
std::string ouro_render_chat(const std::vector<ouro_chat_message> & messages,
                             bool add_generation_prompt, bool enable_thinking);
