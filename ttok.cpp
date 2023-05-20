#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/schriter.h>
#include <unicode/unistr.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <unordered_set>

#include "json.hpp"

using json = nlohmann::json;
const std::string BPE_PRETOK_REGEX =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
class bpe_char_byte_table {
   public:
    bpe_char_byte_table() {
        int n = 0;
        for (uint8_t byte = 0; m_codepoint_to_byte.size() < 256; byte++) {
            bool keep = (byte >= '!' && byte <= '~') ||
                        (byte >= 0xa1 && byte <= 0xac) ||
                        (byte >= 0xae && byte <= 0xff);
            uint32_t codepoint = byte;
            if (!keep) {
                codepoint = 256 + n;
                n++;
            }
            m_byte_to_codepoint[byte] = codepoint;
            m_codepoint_to_byte[codepoint] = byte;
        };
    }
    uint32_t byte_to_codepoint(uint8_t byte) {
        return m_byte_to_codepoint[byte];
    }

    uint8_t codepoint_to_byte(uint32_t codepoint) {
        return m_codepoint_to_byte.at(codepoint);
    }

   private:
    std::array<uint32_t, 256> m_byte_to_codepoint;
    std::unordered_map<uint32_t, uint8_t> m_codepoint_to_byte;
};

typedef std::pair<icu::UnicodeString, icu::UnicodeString> UnicodeBigram;

struct bigram_hash {
    std::size_t operator()(const UnicodeBigram& pair) const {
        return pair.first.hashCode() ^ pair.second.hashCode();
    }
};

void get_bigrams(const std::vector<icu::UnicodeString>& input,
                 std::unordered_set<UnicodeBigram, bigram_hash>& pairs) {
    pairs.clear();
    auto i = input.begin();
    auto prev = *i++;
    for (; i != input.end(); ++i) {
        pairs.insert({prev, *i});
        prev = *i;
    }
}

class BPE {
   public:
    BPE(std::unordered_map<std::string, uint32_t> vocab,
        std::vector<std::string> merges) {
        m_vocab = vocab;
        for (auto pair : vocab) {
            icu::UnicodeString encd = icu::UnicodeString::fromUTF8(pair.first);
            m_reverse_vocab[pair.second] = encd;
        }
        size_t n = 0;
        for (auto merge : merges) {
            std::string s_merge = merge;
            auto spaceidx = s_merge.find(" ");
            auto left =
                icu::UnicodeString::fromUTF8(s_merge.substr(0, spaceidx));
            auto right =
                icu::UnicodeString::fromUTF8(s_merge.substr(spaceidx + 1));
            m_merges[{left, right}] = n++;
        }
    }
    std::vector<uint32_t> encode(const std::string& input) {
        auto normalized = normalize_nfc(input);
        auto pretokenized = pretokenize(normalized);
        std::vector<icu::UnicodeString> tokens_merged;
        for (auto ptok : pretokenized) {
            bpe(ptok, tokens_merged);
        }
        std::vector<uint32_t> final_tokens;
        for (auto mtok : tokens_merged) {
            std::string lookup;
            mtok.toUTF8String(lookup);
            final_tokens.push_back(m_vocab[lookup]);
        }
        return final_tokens;
    }

    std::string decode(const std::vector<uint32_t>& tokens,
                       bool valid_utf8 = true) {
        std::string out;
        for (uint32_t t : tokens) {
            icu::UnicodeString benc = m_reverse_vocab[t];
            icu::StringCharacterIterator schriter(benc);
            for (UChar32 c = schriter.first32(); schriter.hasNext();
                 c = schriter.next32()) {
                out.push_back(m_bs_table.codepoint_to_byte((uint32_t)c));
            }
        }
        // roundtrip through ICU to replace invalid utf8 with U+FFFD
        if (valid_utf8) {
            auto tmp = icu::UnicodeString::fromUTF8(out);
            out.clear();
            tmp.toUTF8String(out);
        }
        return out;
    }
    // https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/bpe.py#L95
    void bpe(icu::UnicodeString token_pretoked,
             std::vector<icu::UnicodeString>& output) {
        if (token_pretoked.length() < 2) {
            output.push_back(token_pretoked);
            return;
        }
        std::vector<icu::UnicodeString> words;
        std::vector<icu::UnicodeString> words_update;
        icu::StringCharacterIterator schriter(token_pretoked);
        UChar32 c;
        for (schriter.setToStart(); schriter.hasNext();) {
            c = schriter.next32PostInc();
            icu::UnicodeString w;
            w.append(c);
            words.push_back(w);
        }
        std::unordered_set<UnicodeBigram, bigram_hash> pairs;
        get_bigrams(words, pairs);
        while (true) {
            size_t min_rank = SIZE_MAX;
            UnicodeBigram to_merge;
            for (auto bigram : pairs) {
                auto loc = m_merges.find(bigram);
                if (loc != m_merges.end() && loc->second < min_rank) {
                    min_rank = loc->second;
                    to_merge = loc->first;
                }
            }
            if (min_rank == SIZE_MAX) {
                break;
            } else {
                auto i = words.begin();
                while (i < words.end()) {
                    if (*i == to_merge.first) {
                        auto inext = i;
                        inext++;
                        if (inext != words.end() && *inext == to_merge.second) {
                            words_update.push_back(*i + *inext);
                            i = inext;
                        } else {
                            words_update.push_back(*i);
                        }
                    } else {
                        words_update.push_back(*i);
                    }
                    ++i;
                }
                words.swap(words_update);
                words_update.clear();
                get_bigrams(words, pairs);
            }
        }
        output.insert(output.end(), words.begin(), words.end());
    }

   private:
    std::unordered_map<std::string, uint32_t> m_vocab;
    std::unordered_map<uint32_t, icu::UnicodeString> m_reverse_vocab;
    std::unordered_map<UnicodeBigram, size_t, bigram_hash> m_merges;
    bpe_char_byte_table m_bs_table;
    std::unique_ptr<icu::RegexPattern> m_pretok_re;

    std::string normalize_nfc(const std::string& input) {
        UErrorCode uerror = U_ZERO_ERROR;
        auto nfcnorm = icu::Normalizer2::getNFCInstance(uerror);
        if (!U_SUCCESS(uerror))
            throw std::runtime_error("could not get ICU NFC normalizer");
        auto icu_ti = icu::UnicodeString::fromUTF8(input);
        std::string out;
        nfcnorm->normalize(icu_ti, uerror).toUTF8String(out);
        if (!U_SUCCESS(uerror))
            throw std::runtime_error("ICU string normalization failed");
        return out;
    }

    std::vector<icu::UnicodeString> pretokenize(const std::string& input) {
        UParseError pe;
        UErrorCode uerror = U_ZERO_ERROR;
        auto bpe_re_icustr = icu::UnicodeString::fromUTF8(BPE_PRETOK_REGEX);
        if (m_pretok_re == nullptr) {
            m_pretok_re = std::unique_ptr<icu::RegexPattern>(
                icu::RegexPattern::compile(bpe_re_icustr, pe, uerror));
            if (!U_SUCCESS(uerror))
                throw std::runtime_error(
                    "Compiling BPE pretokenizer regex failed");
        }
        auto uinput = icu::UnicodeString::fromUTF8(input);
        std::unique_ptr<icu::RegexMatcher> pretok_matcher(
            m_pretok_re->matcher(uinput, uerror));
        std::vector<icu::UnicodeString> pretoks;
        if (!U_SUCCESS(uerror))
            throw std::runtime_error(
                "Creating BPE pretokenizer matcher failed");
        while (pretok_matcher->find()) {
            auto match = pretok_matcher->group(uerror);
            if (!U_SUCCESS(uerror))
                throw std::runtime_error(
                    "Getting BPE pretokenizer regex match failed");
            std::string s;
            icu::UnicodeString out;
            match.toUTF8String(s);
            for (char c : s) {
                uint32_t codepoint = m_bs_table.byte_to_codepoint((uint8_t)c);
                out.append((UChar32)codepoint);
            }
            pretoks.push_back(out);
        }
        return pretoks;
    }
};

int main(int argc, char** argv) {
    // https://huggingface.co/mosaicml/mpt-7b-chat/raw/main/tokenizer.json
    std::ifstream f("mpt-7b-chat-tokenizer.json");
    json tokenizer_config = json::parse(f);
    std::string test_input =
        "Hello, I am a hélpful assistant🤖 and I am here to help!";
    json bpeconfig = tokenizer_config["model"];
    BPE bpe(bpeconfig.at("vocab"), bpeconfig.at("merges"));
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