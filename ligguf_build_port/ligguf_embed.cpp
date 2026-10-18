// LiGGUF embed test harness (Windows/MSVC port)
// usage: ligguf_embed <model.gguf> <texts.txt> <out.bin>
// texts.txt: one UTF-8 string per line; out.bin: N x 512 float32
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

extern "C" int vdb_embed_export(const char* model_fn, const char* text, float* out, int maxdim);

#include "common.h"
model_state g_m;

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: ligguf_embed <model.gguf> <texts.txt> <out.bin>\n");
        return 2;
    }
    const char* model = argv[1];
    const char* intxt = argv[2];
    const char* outfn = argv[3];

    FILE* f = fopen(intxt, "rb");
    if (!f) { fprintf(stderr, "can't open %s\n", intxt); return 3; }
    // read all, split lines
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> buf(sz + 1, 0);
    fread(buf.data(), 1, sz, f);
    fclose(f);

    std::vector<std::string> lines;
    std::string cur;
    for (char c : buf) {
        if (c == '\n') { lines.push_back(cur); cur.clear(); }
        else if (c != '\r') cur.push_back(c);
    }
    if (!cur.empty()) lines.push_back(cur);

    const int DIM = 512;
    std::vector<float> all;
    int ok = 0, fail = 0;
    for (const std::string& t : lines) {
        if (t.empty()) { continue; }
        std::vector<float> emb(DIM, 0.f);
        int rc = vdb_embed_export(model, t.c_str(), emb.data(), DIM);
        if (rc == 0) { all.insert(all.end(), emb.begin(), emb.end()); ok++; }
        else { fprintf(stderr, "embed fail(%d) text='%s'\n", rc, t.substr(0, 30).c_str()); fail++; }
    }

    FILE* o = fopen(outfn, "wb");
    if (!o) { fprintf(stderr, "can't write %s\n", outfn); return 4; }
    fwrite(&ok, sizeof(int), 1, o);
    fwrite(all.data(), sizeof(float), all.size(), o);
    fclose(o);

    fprintf(stderr, "embedded=%d fail=%d dim=%d\n", ok, fail, DIM);
    return 0;
}
