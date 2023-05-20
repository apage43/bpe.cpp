#include "json.hpp"
#include <unicode/unistr.h>
#include <unicode/normalizer2.h>
#include <unicode/schriter.h>
#include <unicode/regex.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <regex>
#include <unordered_set>

using json = nlohmann::json;
const std::string BPE_PRETOK_REGEX = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
struct bpe_char_byte_table
{
    std::array<uint32_t, 256> byte_to_codepoint;
    std::unordered_map<uint32_t, uint8_t> codepoint_to_byte;
    bpe_char_byte_table()
    {
        int n = 0;
        for (uint8_t byte = 0; codepoint_to_byte.size() < 256; byte++)
        {
            bool keep = (byte >= '!' && byte <= '~') || (byte >= 0xa1 && byte <= 0xac) || (byte >= 0xae && byte <= 0xff);
            uint32_t codepoint = byte;
            if (!keep)
            {
                codepoint = 256 + n;
                n++;
            }
            byte_to_codepoint[byte] = codepoint;
            codepoint_to_byte[codepoint] = byte;
        };
    }
};

const static bpe_char_byte_table bs_table;
typedef std::pair<icu::UnicodeString, icu::UnicodeString> UnicodeBigram;

uint32_t byte_to_codepoint(uint8_t byte)
{
    return bs_table.byte_to_codepoint[byte];
}

uint8_t codepoint_to_byte(uint32_t codepoint)
{
    return bs_table.codepoint_to_byte.at(codepoint);
}

std::string normalize_nfc(const std::string &input)
{
    UErrorCode uerror = U_ZERO_ERROR;
    auto nfcnorm = icu::Normalizer2::getNFCInstance(uerror);
    if (!U_SUCCESS(uerror))
        throw;
    auto icu_ti = icu::UnicodeString::fromUTF8(input);
    std::string out;
    nfcnorm->normalize(icu_ti, uerror).toUTF8String(out);
    if (!U_SUCCESS(uerror))
        throw;
    return out;
}

std::vector<icu::UnicodeString> pretokenize(const std::string &input)
{
    UParseError pe;
    UErrorCode uerror = U_ZERO_ERROR;
    auto bpe_re_icustr = icu::UnicodeString::fromUTF8(BPE_PRETOK_REGEX);
    std::unique_ptr<icu::RegexPattern> pretok_re(icu::RegexPattern::compile(bpe_re_icustr, pe, uerror));
    auto uinput = icu::UnicodeString::fromUTF8(input);
    std::unique_ptr<icu::RegexMatcher> pretok_matcher(pretok_re->matcher(uinput, uerror));
    std::vector<icu::UnicodeString> pretoks;
    while (pretok_matcher->find())
    {
        auto match = pretok_matcher->group(uerror);
        std::string s;
        icu::UnicodeString out;
        match.toUTF8String(s);
        for (char c : s)
        {
            uint32_t codepoint = byte_to_codepoint((uint8_t)c);
            out.append((UChar32)codepoint);
        }
        pretoks.push_back(out);
    }
    return pretoks;
}

struct bigram_hash
{
    std::size_t operator()(const UnicodeBigram &pair) const
    {
        return pair.first.hashCode() ^ pair.second.hashCode();
    }
};

void get_bigrams(const std::vector<icu::UnicodeString> &input, std::unordered_set<UnicodeBigram, bigram_hash> &pairs)
{
    pairs.clear();
    auto i = input.begin();
    auto prev = *i++;
    for (; i != input.end(); ++i)
    {
        pairs.insert({prev, *i});
        prev = *i;
    }
}

// https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/bpe.py#L95
struct BPE
{
    std::unordered_map<std::string, uint32_t> vocab;
    std::unordered_map<uint32_t, std::string> reverse_vocab;
    std::unordered_map<UnicodeBigram, size_t, bigram_hash> merges;

    void bpe(icu::UnicodeString token_pretoked, std::vector<icu::UnicodeString> &output)
    {
        if (token_pretoked.length() < 2)
        {
            output.push_back(token_pretoked);
            return;
        }
        std::vector<icu::UnicodeString> words;
        std::vector<icu::UnicodeString> words_update;
        icu::StringCharacterIterator schriter(token_pretoked);
        UChar32 c;
        for (schriter.setToStart(); schriter.hasNext();)
        {
            c = schriter.next32PostInc();
            icu::UnicodeString w;
            w.append(c);
            words.push_back(w);
        }
        std::unordered_set<UnicodeBigram, bigram_hash> pairs;
        get_bigrams(words, pairs);
        // for(auto bigram: pairs) {
        //     std::string l,r;
        //     bigram.first.toUTF8String(l);
        //     bigram.second.toUTF8String(r);
        //     std::cerr << "bigram: " << l << "," << r << std::endl;
        // }
        while (true)
        {
            size_t min_rank = SIZE_MAX;
            UnicodeBigram to_merge;
            for (auto bigram : pairs)
            {
                auto loc = merges.find(bigram);
                if (loc != merges.end() && loc->second < min_rank)
                {
                    min_rank = loc->second;
                    to_merge = loc->first;
                }
            }
            if (min_rank == SIZE_MAX)
            {
                // std::cerr << "no merges to do" << std::endl;
                break;
            }
            else
            {
                // {
                //     std::string l, r;
                //     to_merge.first.toUTF8String(l);
                //     to_merge.second.toUTF8String(r);
                //     std::cerr << "merging bigram: " << l << "," << r << std::endl;
                // }
                auto i = words.begin();
                while (i < words.end())
                {
                    if (*i == to_merge.first)
                    {
                        auto inext = i;
                        inext++;
                        if (inext != words.end() && *inext == to_merge.second)
                        {
                            words_update.push_back(*i + *inext);
                            i = inext;
                        }
                        else
                        {
                            words_update.push_back(*i);
                        }
                    }
                    else
                    {
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
};

void from_json(const json &j, BPE &bpe)
{
    j.at("vocab").get_to(bpe.vocab);
    bpe.reverse_vocab.clear();
    for (auto pair : bpe.vocab)
    {
        bpe.reverse_vocab[pair.second] = pair.first;
    }
    size_t n = 0;
    for (auto merge : j.at("merges"))
    {
        std::string s_merge = merge;
        auto spaceidx = s_merge.find(" ");
        auto left = icu::UnicodeString::fromUTF8(s_merge.substr(0, spaceidx));
        auto right = icu::UnicodeString::fromUTF8(s_merge.substr(spaceidx + 1));
        bpe.merges[{left, right}] = n++;
    }
}

int main(int argc, char **argv)
{
    // https://huggingface.co/mosaicml/mpt-7b-chat/raw/main/tokenizer.json
    std::ifstream f("mpt-7b-chat-tokenizer.json");
    json tokenizer_config = json::parse(f);
    std::string test_input = "Hello, I am a hélpful assistant🤖 and I am here to help!";
    auto normalized = normalize_nfc(test_input);
    auto pretokenized = pretokenize(normalized);
    BPE bpe = tokenizer_config["model"].get<BPE>();
    std::vector<icu::UnicodeString> tokens_merged;
    for (auto ptok : pretokenized)
    {
        bpe.bpe(ptok, tokens_merged);
    }
    std::cerr << "merged: ";
    std::vector<uint32_t> final_tokens;
    for (auto mtok : tokens_merged)
    {
        std::string lookup;
        mtok.toUTF8String(lookup);
        final_tokens.push_back(bpe.vocab[lookup]);
        std::cerr << lookup << "|";
    }
    std::cerr << std::endl;
    std::cerr << "tokens: [";
    for (auto tok : final_tokens)
    {
        std::cerr << tok << ", ";
    }
    std::cerr << std::endl;
    return 0;
}