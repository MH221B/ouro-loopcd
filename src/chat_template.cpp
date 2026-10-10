#include "chat_template.h"

std::string ouro_render_chat(const std::vector<ouro_chat_message> & messages,
                             bool add_generation_prompt, bool enable_thinking) {
    std::string out;

    if (!messages.empty() && messages[0].role == "system") {
        out += "<|im_start|>system\n" + messages[0].content + "<|im_end|>\n";
    } else {
        out += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    }

    for (size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].role == "system" && i == 0) continue;   // already rendered above
        out += "<|im_start|>" + messages[i].role + "\n" + messages[i].content + "<|im_end|>\n";
    }

    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        if (enable_thinking) out += "<think>\n";
    }
    return out;
}
