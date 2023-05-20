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
    json bpeconfig = tokenizer_config["model"];

    bpecpp::BPE bpe(bpeconfig.at("vocab"), bpeconfig.at("merges"));

    std::string test_input =
        "Hello, I am a hélpful assistant🤖 and I am here to help!";

    std::vector<uint32_t> final_tokens = bpe.encode(test_input);

    std::cerr << "input: " << test_input << std::endl;
    std::cerr << "tokens: ";
    for (auto tok = final_tokens.begin(); tok != final_tokens.end();) {
        std::cerr << *tok++;
        if (tok != final_tokens.end())
            std::cerr << ", ";
    }
    std::cerr << std::endl;
    std::cerr << "decoded: " << bpe.decode(final_tokens) << std::endl;

    std::cerr << "test invalid utf8 ending" << std::endl;
    final_tokens.resize(11);
    std::cerr << "decoded: " << bpe.decode(final_tokens) << std::endl;
    return 0;
}