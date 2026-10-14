// Development tool for the native Qwen engine: feed token IDs, print greedy tokens and timings.
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

// Greedy tokens after `logits`, `count` of them; with mtp > 0 through draft rounds.
// A draft is kept while it equals the model's own choice, so the tokens do not
// depend on mtp. The first token is not consumed.
static std::vector<int32_t> generate(wq::Engine& engine, const float* logits, int count, int mtp,
                                     int* rounds = nullptr, int* drafted = nullptr, int* accepted_total = nullptr) {
  int vocab = engine.vocab();
  std::vector<int32_t> out;
  int token = argmax(logits, vocab);
  if (mtp > 0 && engine.has_mtp()) {
    while ((int)out.size() < count) {
      std::vector<int32_t> round{token};
      int want = std::min(mtp, count - (int)out.size() - 1);
      if (want == 0) { out.push_back(token); break; }
      for (int i = 0; i < want; ++i) round.push_back(argmax(engine.draft(round.back()), vocab));
      const float* rows = engine.verify(round.data(), (int)round.size());
      int accepted = 0;
      while (accepted < want && argmax(rows + (size_t)accepted * vocab, vocab) == round[accepted + 1]) ++accepted;
      int next = argmax(rows + (size_t)accepted * vocab, vocab);
      engine.commit(accepted + 1);
      out.insert(out.end(), round.begin(), round.begin() + accepted + 1);
      if (rounds != nullptr) { ++*rounds; *drafted += want; *accepted_total += accepted; }
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

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr,
            "usage: wq-cli MODEL_DIR TOKENS [--greedy N] [--mtp N] [--8bit] [--no-seed] [--chunk N] [--rows N] [--slots N] [--dump DIR] [--logits FILE]\n"
            "  TOKENS   comma-separated token IDs, or @FILE holding them\n"
            "  --chunk  tokens per eval call for the prompt (default 1)\n"
            "  --rows   prompt rows processed together inside one layer (default 1024)\n"
            "  --mtp    draft this many tokens per round with the draft layer (default 0: off)\n"
            "  --state-check N   prompt cache check: snapshot after N prompt tokens, then compare the\n"
            "                    rest of the prompt and the greedy tokens with and without restoring\n"
            "  --state-file PATH also save and load the snapshot through this file\n"
            "  --load-state PATH start from a saved state; TOKENS then continue it\n"
            "  --save-state PATH save the state after the prompt\n"
            "  --image PATCHES --grid H,W --vision-catalog QwenVision.json\n"
            "                    one image as processor patches (float32 [H*W, 1536]); TOKENS hold its tokens\n"
            "  --features-out PATH write the image encoder output (float32)\n"
            "  --features-in PATH  use these image features instead of encoding the patches\n");
    return 2;
  }
  wq::Options options;
  int greedy = 0, chunk = 1, mtp = 0, state_check = 0;
  std::string state_file, save_state, load_state, image_path, features_out, features_in;
  int grid_h = 0, grid_w = 0;
  std::string logits_path;
  for (int i = 3; i < argc; ++i) {
    std::string flag = argv[i];
    const char* value = i + 1 < argc ? argv[i + 1] : "";
    if (flag == "--greedy") { greedy = atoi(value); ++i; }
    else if (flag == "--chunk") { chunk = atoi(value); ++i; }
    else if (flag == "--slots") { options.slots = atoi(value); ++i; }
    else if (flag == "--rows") { options.chunk = atoi(value); ++i; }
    else if (flag == "--mtp") { mtp = atoi(value); ++i; }
    else if (flag == "--no-seed") { options.seed = false; }
    else if (flag == "--8bit") { options.eight_bit = true; }
    else if (flag == "--state-check") { state_check = atoi(value); ++i; }
    else if (flag == "--state-file") { state_file = value; ++i; }
    else if (flag == "--save-state") { save_state = value; ++i; }
    else if (flag == "--image") { image_path = value; ++i; }
    else if (flag == "--grid") { sscanf(value, "%d,%d", &grid_h, &grid_w); ++i; }
    else if (flag == "--vision-catalog") { options.vision_catalog = value; ++i; }
    else if (flag == "--features-out") { features_out = value; ++i; }
    else if (flag == "--features-in") { features_in = value; ++i; }
    else if (flag == "--load-state") { load_state = value; ++i; }
    else if (flag == "--dump") { options.dump_dir = value; ++i; }
    else if (flag == "--logits") { logits_path = value; ++i; }
    else { fprintf(stderr, "unknown option %s\n", flag.c_str()); return 2; }
  }
  std::string text = argv[2];
  if (!text.empty() && text[0] == '@') {
    FILE* file = fopen(text.c_str() + 1, "rb");
    if (file == nullptr) { fprintf(stderr, "cannot read %s\n", text.c_str() + 1); return 2; }
    text.clear();
    char buffer[65536];
    size_t got;
    while ((got = fread(buffer, 1, sizeof(buffer), file)) > 0) text.append(buffer, got);
    fclose(file);
  }
  std::vector<int32_t> prompt;
  std::stringstream stream(text);
  for (std::string item; std::getline(stream, item, ',');)
    if (!item.empty()) prompt.push_back(atoi(item.c_str()));
  if (prompt.empty() || chunk < 1) { fprintf(stderr, "no tokens\n"); return 2; }

  try {
    double started = now();
    auto engine = wq::Engine::open(argv[1], options);
    fprintf(stderr, "load %.2fs\n", now() - started);

    if (state_check > 0) {
      if (state_check >= (int)prompt.size()) { fprintf(stderr, "--state-check must be shorter than the prompt\n"); return 2; }
      int vocab = engine->vocab();
      auto continuation = [&](std::vector<float>& last, std::vector<int32_t>& out) {
        const float* l = engine->eval(prompt.data() + state_check, (int)prompt.size() - state_check, 1);
        last.assign(l, l + vocab);
        out = generate(*engine, l, std::max(greedy, 1), mtp);
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
      double restored = now();
      engine->restore(*state);
      fprintf(stderr, "restore %.3fs\n", now() - restored);
      continuation(again, again_tokens);
      compare("after restore");
      if (!state_file.empty()) {
        double saved = now();
        engine->save(*state, state_file);
        auto loaded = engine->load(state_file);
        fprintf(stderr, "save and load %.3fs\n", now() - saved);
        engine->reset();
        engine->restore(*loaded);
        continuation(again, again_tokens);
        compare("after save, reset and load");
      }
      return ok ? 0 : 1;
    }

    if (!load_state.empty()) {
      auto state = engine->load(load_state);
      engine->restore(*state);
      fprintf(stderr, "restored %zu tokens\n", state->tokens.size());
    }
    if (!image_path.empty()) {
      wq::ImageInput image;
      image.grid_h = grid_h;
      image.grid_w = grid_w;
      image.patches.resize((size_t)grid_h * grid_w * 1536);
      FILE* file = fopen(image_path.c_str(), "rb");
      if (file == nullptr || fread(image.patches.data(), 4, image.patches.size(), file) != image.patches.size()) {
        fprintf(stderr, "cannot read %zu floats from %s\n", image.patches.size(), image_path.c_str());
        return 2;
      }
      fclose(file);
      double encoded = now();
      if (!features_out.empty()) {
        std::vector<float> features = engine->encode_image(image.patches.data(), grid_h, grid_w);
        fprintf(stderr, "encode %.3fs\n", now() - encoded);
        FILE* out = fopen(features_out.c_str(), "wb");
        fwrite(features.data(), 4, features.size(), out);
        fclose(out);
      }
      if (!features_in.empty()) {
        image.features.resize((size_t)grid_h * grid_w / 4 * 2560);
        FILE* in = fopen(features_in.c_str(), "rb");
        if (in == nullptr || fread(image.features.data(), 4, image.features.size(), in) != image.features.size()) {
          fprintf(stderr, "cannot read the image features from %s\n", features_in.c_str());
          return 2;
        }
        fclose(in);
      }
      started = now();
      const float* logits = engine->eval_images(prompt.data(), (int)prompt.size(), {image}, 1);
      fprintf(stderr, "image prompt %zu tokens %.3fs\n", prompt.size(), now() - started);
      if (!logits_path.empty()) {
        FILE* out = fopen(logits_path.c_str(), "wb");
        fwrite(logits, 4, engine->vocab(), out);
        fclose(out);
      }
      std::vector<int32_t> tokens = generate(*engine, logits, greedy, mtp);
      printf("greedy");
      for (int32_t token : tokens) printf(" %d", token);
      printf("\n");
      return 0;
    }
    FILE* logits_file = logits_path.empty() ? nullptr : fopen(logits_path.c_str(), "wb");
    started = now();
    const float* logits = nullptr;
    for (size_t at = 0; at < prompt.size(); at += chunk) {
      int count = (int)std::min<size_t>(chunk, prompt.size() - at);
      logits = engine->eval(prompt.data() + at, count, 1);
      if (logits_file != nullptr) fwrite(logits, 4, engine->vocab(), logits_file);
      if (chunk == 1) printf("prompt %zu token %d argmax %d\n", at, prompt[at], argmax(logits, engine->vocab()));
    }
    double prefill = now() - started;
    fprintf(stderr, "prefill %zu tokens %.3fs %.1f tok/s\n%s\n", prompt.size(), prefill, prompt.size() / prefill, engine->stats().c_str());
    if (logits_file != nullptr) fclose(logits_file);

    if (!save_state.empty()) engine->save(*engine->snapshot(), save_state);
    int drafted = 0, accepted_total = 0, rounds = 0;
    started = now();
    std::vector<int32_t> tokens = generate(*engine, logits, greedy, mtp, &rounds, &drafted, &accepted_total);
    printf("greedy");
    for (int32_t token : tokens) printf(" %d", token);
    printf("\n");
    if (rounds > 0) fprintf(stderr, "mtp rounds %d drafted %d accepted %d\n", rounds, drafted, accepted_total);
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
