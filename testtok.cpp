#include <cstdint>
#include <fstream>
#include <iostream>

#include "bpe.h"
#include "json.hpp"

using json = nlohmann::json;

int main(int argc, char** argv) {
    // https://huggingface.co/mosaicml/mpt-7b-chat/raw/main/tokenizer.json
    std::ifstream f("../mpt-7b-chat-tokenizer.json");
    json tokenizer_config = json::parse(f);

    std::vector<bpecpp::additional_vocab_item> added_vocab;
    for (auto javi : tokenizer_config.at("added_tokens")) {
        added_vocab.push_back({.id = javi.at("id"),
                               .content = javi.at("content"),
                               .special = javi.at("special")});
    }
    bpecpp::AdditionalVocabAdapter av(added_vocab);

    json bpeconfig = tokenizer_config.at("model");
    bpecpp::BPE bpe(bpeconfig.at("vocab"), bpeconfig.at("merges"));

    std::string test_input = "<|im_start|>system\nyou're a helpful AI assistant 🤖 that likes emojis<|im_end|>";

    std::vector<uint32_t> final_tokens = av.encode(test_input, bpe);

    std::cerr << "input: " << test_input << std::endl;
    std::cerr << "tokens: ";
    for (auto tok = final_tokens.begin(); tok != final_tokens.end();) {
        std::cerr << *tok++;
        if (tok != final_tokens.end())
            std::cerr << ", ";
    }
    std::cerr << std::endl;
    std::cerr << "decoded: " << av.decode(final_tokens, bpe) << std::endl;

    std::cerr << "test invalid utf8 ending" << std::endl;
    final_tokens.resize(11);
    std::cerr << "decoded: " << av.decode(final_tokens, bpe) << std::endl;
    return 0;
}