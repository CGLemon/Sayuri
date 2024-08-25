#pragma once
#include <cstdint>
#include <cstring>
#include <string>

#define SHA256_BLOCK_SIZE 32
#define SHA256_BYTE std::uint8_t
#define SHA256_WORD std::uint32_t

struct SHA256_CTX {
    SHA256_BYTE data[64];
    SHA256_WORD state[8];
    size_t datalen;
    size_t bitlen;
};

#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

static constexpr SHA256_WORD Sha256_k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

inline void sha256_transform(SHA256_CTX *ctx, const SHA256_BYTE *data) {
    size_t i, j;
    SHA256_WORD m[64], mv[8];

    for (i = 0, j = 0; i < 16; ++i, j += 4) {
        m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
    }
    for ( ; i < 64; ++i) {
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
    }
    for (i = 0; i < 8; ++i) {
        mv[i] = ctx->state[i];
    }
    for (i = 0; i < 64; ++i) {
        SHA256_WORD t1 = mv[7] + EP1(mv[4]) + CH(mv[4],mv[5],mv[6]) + Sha256_k[i] + m[i];
        SHA256_WORD t2 = EP0(mv[0]) + MAJ(mv[0],mv[1],mv[2]);
        mv[7] = mv[6];
        mv[6] = mv[5];
        mv[5] = mv[4];
        mv[4] = mv[3] + t1;
        mv[3] = mv[2];
        mv[2] = mv[1];
        mv[1] = mv[0];
        mv[0] = t1 + t2;
    }
    for (i = 0; i < 8; ++i) {
        ctx->state[i] += mv[i];
    }
}

inline void sha256_init(SHA256_CTX *ctx) {
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
}

inline void sha256_update(SHA256_CTX *ctx, const SHA256_BYTE *data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 64) {
            sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

inline void sha256_final(SHA256_CTX *ctx, SHA256_BYTE *hash) {
    // Pad whatever data is left in the buffer.
    size_t end = ctx->datalen;
    if (ctx->datalen < 56) {
        ctx->data[end++] = 0x80;
        while (end < 56) {
            ctx->data[end++] = 0x00;
        }
    } else {
        ctx->data[end++] = 0x80;
        while (end < 64) {
            ctx->data[end++] = 0x00;
        }
        sha256_transform(ctx, ctx->data);
        std::memset(ctx->data, 0, 56);
    }

    // Append to the padding the total message's length in bits and transform.
    ctx->bitlen += ctx->datalen * 8;
    for (size_t j = 0; j < 8 ; ++j) {
        ctx->data[63 - j] = ctx->bitlen >> (8 * j);
    }
    sha256_transform(ctx, ctx->data);

    // Since this implementation uses little endian byte ordering and SHA uses big endian,
    // reverse all the bytes when copying the final state to the output hash.
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 8 ; ++j) {
            hash[i + 4 * j]  = (ctx->state[j] >> (24 - i * 8)) & 0x000000ff;
        }
    }
}

inline std::string GetSha256Digest(std::string &text) {
    SHA256_CTX ctx;
    SHA256_BYTE digest[SHA256_BLOCK_SIZE];
    auto text_ptr = reinterpret_cast<const unsigned char*>(text.c_str());

    sha256_init(&ctx);
    sha256_update(&ctx, text_ptr, text.size());
    sha256_final(&ctx, digest);

    char buf[2*SHA256_BLOCK_SIZE+1] = { 0 };
    for (int i = 0; i < SHA256_BLOCK_SIZE; ++i) {
        std::sprintf(buf+i*2, "%02X", digest[i]);
    }
    return std::string(buf);
}

#undef ROTLEFT
#undef ROTRIGHT
#undef CH
#undef MAJ
#undef EP0
#undef EP1
#undef SIG0
#undef SIG1
