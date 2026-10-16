// Development tool for the native DeepSeek V4.1 engine: feed token IDs, print greedy tokens and timings.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

#include "engine.h"

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int argmax(const float* row, int count) {
  int best = 0;
  for (int i = 1; i < count; ++i)
    if (row[i] > row[best]) best = i;
  return best;
}

struct Rounds { int rounds = 0, drafted = 0, accepted = 0, skipped = 0; double draft_seconds = 0, verify_seconds = 0; };

// After two drafted rounds in a row without an accepted draft, skip drafting for 1, then 2,
// 4 ... 16 tokens; an accepted draft resets the wait (as the Python runtime does).
struct Backoff {
  int skip = 0, wait = 1, streak = 0;
  bool enabled = true;
  bool draft_now() {
    if (!enabled || skip == 0) return true;
    --skip;
    return false;
  }
  void observe(int accepted) {
    if (accepted > 0) { streak = 0; wait = 1; return; }
    if (++streak < 2) return;
    skip = wait;
    wait = std::min(wait * 2, 16);
    streak = 0;
  }
};

// Greedy tokens after `logits`, `count` of them; the last one is not consumed. With
// `dspark`, through draft rounds: a draft is kept while its confidence reaches
// `threshold` (the App's default for V4.1), and accepted while it equals the model's
// own choice, so the tokens do not depend on DSpark.
static std::vector<int32_t> generate(wd::Engine& engine, const float* logits, int count, bool dspark = false,
                                     float threshold = 0.75f, Rounds* stats = nullptr, bool backoff = true) {
  Backoff policy;
  policy.enabled = backoff;
  int vocab = engine.vocab();
  std::vector<int32_t> out;
  int token = argmax(logits, vocab);
  if (dspark && engine.has_dspark()) {
    while ((int)out.size() < count) {
      int budget = std::min(engine.draft_block(), count - (int)out.size() - 1);
      std::vector<int32_t> round{token};
      double started = now();
      bool drafting = budget > 0 && policy.draft_now();
      if (!drafting && budget > 0 && stats != nullptr) ++stats->skipped;
      if (drafting) {
        engine.draft(token);
        for (int i = 0; i < budget; ++i) {
          float confidence = 0;
          const float* row = engine.draft_step(i, round.back(), &confidence);
          if (confidence < threshold) break;
          round.push_back(argmax(row, vocab));
        }
      }
      double drafted = now();
      const float* rows = engine.verify(round.data(), (int)round.size());
      int accepted = 0;
      while (accepted + 1 < (int)round.size() && argmax(rows + (size_t)accepted * vocab, vocab) == round[accepted + 1]) ++accepted;
      int next = argmax(rows + (size_t)accepted * vocab, vocab);
      engine.commit(accepted + 1);
      if (drafting) policy.observe(accepted);
      out.insert(out.end(), round.begin(), round.begin() + accepted + 1);
      if (stats != nullptr) {
        ++stats->rounds;
        stats->drafted += (int)round.size() - 1;
        stats->accepted += accepted;
        stats->draft_seconds += drafted - started;
        stats->verify_seconds += now() - drafted;
      }
      token = next;
    }
    out.resize(std::min<size_t>(out.size(), count));
    return out;
  }
  for (int i = 0; i < count; ++i) {
    out.push_back(token);
    if (i + 1 < count) token = argmax(engine.eval(&token, 1, 1), vocab);
  }
  return out;
}

static bool read_file(const std::string& path, std::string& text) {
  FILE* file = fopen(path.c_str(), "rb");
  if (file == nullptr) return false;
  char buffer[65536];
  size_t got;
  while ((got = fread(buffer, 1, sizeof(buffer), file)) > 0) text.append(buffer, got);
  fclose(file);
  return true;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr,
            "usage: wd-cli MODEL_DIR TOKENS --token-map FILE [--greedy N] [--chunk N] [--rows N] [--slots N] [--no-seed]\n"
            "              [--dump DIR] [--logits FILE] [--state-check N [--state-file PATH]] [--dspark [--threshold X]]\n"
            "  TOKENS       comma-separated token IDs, or @FILE holding them\n"
            "  --token-map  int32 compressed Engram id per vocabulary id (Scripts/deepseek_v41_reference.py)\n"
            "  --chunk      tokens per eval call for the prompt (default: the whole prompt)\n"
            "  --rows       prompt rows whose attention runs together inside one layer (default 1024)\n"
            "  --dspark     decode through DSpark draft rounds (greedy; drafts kept from confidence --threshold, 0.75)\n"
            "  --verify-check N verify(N tokens) must give the logits of N one-token evals, bit for bit\n"
            "  --state-check N  prompt cache check: snapshot after N prompt tokens, then compare the\n"
            "                   rest of the prompt and the greedy tokens with and without restoring\n");
    return 2;
  }
  wd::Options options;
  int greedy = 0, chunk = 0, state_check = 0, verify_check = 0;
  bool dspark = false, backoff = true;
  float threshold = 0.75f;
  std::string token_map, logits_path, state_file;
  for (int i = 3; i < argc; ++i) {
    std::string flag = argv[i];
    const char* value = i + 1 < argc ? argv[i + 1] : "";
    if (flag == "--greedy") { greedy = atoi(value); ++i; }
    else if (flag == "--chunk") { chunk = atoi(value); ++i; }
    else if (flag == "--rows") { options.chunk = atoi(value); ++i; }
    else if (flag == "--slots") { options.slots = atoi(value); ++i; }
    else if (flag == "--threads") { options.read_threads = atoi(value); ++i; }
    else if (flag == "--no-seed") { options.seed = false; }
    else if (flag == "--dspark") { dspark = true; options.dspark = true; }
    else if (flag == "--threshold") { threshold = (float)atof(value); ++i; }
    else if (flag == "--no-backoff") { backoff = false; }
    else if (flag == "--token-map") { token_map = value; ++i; }
    else if (flag == "--dump") { options.dump_dir = value; ++i; }
    else if (flag == "--logits") { logits_path = value; ++i; }
    else if (flag == "--state-check") { state_check = atoi(value); ++i; }
    else if (flag == "--state-file") { state_file = value; ++i; }
    else if (flag == "--verify-check") { verify_check = atoi(value); ++i; }
    else { fprintf(stderr, "unknown option %s\n", flag.c_str()); return 2; }
  }
  std::string text = argv[2];
  if (!text.empty() && text[0] == '@') {
    std::string path = text.substr(1);
    text.clear();
    if (!read_file(path, text)) { fprintf(stderr, "cannot read %s\n", path.c_str()); return 2; }
  }
  std::vector<int32_t> prompt;
  std::stringstream stream(text);
  for (std::string item; std::getline(stream, item, ',');)
    if (!item.empty()) prompt.push_back(atoi(item.c_str()));
  std::string map_bytes;
  if (token_map.empty() || !read_file(token_map, map_bytes) || map_bytes.size() % 4 != 0) {
    fprintf(stderr, "--token-map FILE is required\n");
    return 2;
  }
  options.token_map.resize(map_bytes.size() / 4);
  memcpy(options.token_map.data(), map_bytes.data(), map_bytes.size());
  if (prompt.empty()) { fprintf(stderr, "no tokens\n"); return 2; }
  if (chunk < 1) chunk = (int)prompt.size();

  try {
    double started = now();
    auto engine = wd::Engine::open(argv[1], options);
    fprintf(stderr, "load %.2fs\n", now() - started);
    int vocab = engine->vocab();

    if (state_check > 0) {
      if (state_check >= (int)prompt.size()) { fprintf(stderr, "--state-check must be shorter than the prompt\n"); return 2; }
      auto continuation = [&](std::vector<float>& last, std::vector<int32_t>& out) {
        const float* l = engine->eval(prompt.data() + state_check, (int)prompt.size() - state_check, 1);
        last.assign(l, l + vocab);
        out = generate(*engine, l, std::max(greedy, 1), dspark, threshold);
      };
      engine->eval(prompt.data(), state_check, 1);
      double taken = now();
      auto state = engine->snapshot();
      fprintf(stderr, "snapshot %zu tokens %.1f MB in %.3fs\n", state->tokens.size(), state->data.size() / 1e6, now() - taken);
      std::vector<float> first, again;
      std::vector<int32_t> first_tokens, again_tokens;
      continuation(first, first_tokens);
      bool ok = true;
      auto compare = [&](const char* name) {
        bool same = first == again && first_tokens == again_tokens;
        printf("%s %s\n", same ? "same" : "DIFFERENT", name);
        ok = ok && same;
      };
      engine->restore(*state);
      continuation(again, again_tokens);
      compare("after restore");
      if (!state_file.empty()) {
        engine->save(*state, state_file);
        auto loaded = engine->load(state_file);
        engine->reset();
        engine->restore(*loaded);
        continuation(again, again_tokens);
        compare("after save, reset and load");
      }
      printf("greedy");
      for (int32_t token : first_tokens) printf(" %d", token);
      printf("\n");
      return ok ? 0 : 1;
    }

    if (verify_check > 0) {
      if (verify_check > 8) { fprintf(stderr, "--verify-check takes 1 to 8\n"); return 2; }
      const float* l = engine->eval(prompt.data(), (int)prompt.size(), 1);
      auto state = engine->snapshot();
      std::vector<int32_t> tokens{argmax(l, vocab)};
      std::vector<float> single;
      for (int i = 0; i < verify_check; ++i) {
        const float* row = engine->eval(&tokens.back(), 1, 1);
        single.insert(single.end(), row, row + vocab);
        if (i + 1 < verify_check) tokens.push_back(argmax(row, vocab));
      }
      engine->restore(*state);
      const float* rows = engine->verify(tokens.data(), verify_check);
      int differing = 0;
      for (int i = 0; i < verify_check; ++i)
        if (memcmp(rows + (size_t)i * vocab, single.data() + (size_t)i * vocab, (size_t)vocab * 4) != 0) ++differing;
      engine->commit(verify_check);
      printf("%s verify of %d tokens: %d rows differ from one-token evals\n", differing ? "DIFFERENT" : "same", verify_check, differing);
      return differing ? 1 : 0;
    }

    FILE* logits_file = logits_path.empty() ? nullptr : fopen(logits_path.c_str(), "wb");
    started = now();
    const float* logits = nullptr;
    for (size_t at = 0; at < prompt.size(); at += chunk) {
      int count = (int)std::min<size_t>(chunk, prompt.size() - at);
      logits = engine->eval(prompt.data() + at, count, 1);
    }
    if (logits_file != nullptr) {
      fwrite(logits, 4, vocab, logits_file);
      fclose(logits_file);
    }
    double prefill = now() - started;
    fprintf(stderr, "prefill %zu tokens %.3fs %.1f tok/s\n%s\n", prompt.size(), prefill, prompt.size() / prefill, engine->stats().c_str());
    started = now();
    Rounds rounds;
    std::vector<int32_t> tokens = generate(*engine, logits, greedy, dspark, threshold, &rounds, backoff);
    printf("greedy");
    for (int32_t token : tokens) printf(" %d", token);
    printf("\n");
    if (rounds.rounds > 0)
      fprintf(stderr, "dspark rounds %d skipped %d drafted %d accepted %d draft %.3fs verify %.3fs\n", rounds.rounds,
              rounds.skipped, rounds.drafted, rounds.accepted, rounds.draft_seconds, rounds.verify_seconds);
    if (greedy > 1) {
      double decode = now() - started;
      fprintf(stderr, "decode %d tokens %.3fs %.2f tok/s\n", greedy - 1, decode, (greedy - 1) / decode);
    }
    fprintf(stderr, "%s\n", engine->stats().c_str());
  } catch (const std::exception& error) {
    fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
  return 0;
}
