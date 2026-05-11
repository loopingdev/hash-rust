#include <stdint.h>
#include <string.h>

__constant__ uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ void keccak_f(uint64_t state[25]) {
    int ro[24]   = {1,62,28,27,36,44,6,55,20,3,10,43,25,39,41,45,15,21,8,18,2,61,56,14};
    int pi_x[24] = {1,6,9,22,14,20,2,12,13,19,23,15,4,24,21,8,16,5,3,18,17,11,7,10};

    for (int r = 0; r < 24; r++) {
        uint64_t C[5], D[5];

        for (int x = 0; x < 5; x++)
            C[x] = state[x] ^ state[x+5] ^ state[x+10] ^ state[x+15] ^ state[x+20];

        for (int x = 0; x < 5; x++)
            D[x] = C[(x+4)%5] ^ rotl64(C[(x+1)%5], 1);

        for (int i = 0; i < 25; i++)
            state[i] ^= D[i%5];

        uint64_t B[25];
        B[0] = state[0];

        uint64_t cur = state[1];

        for (int t = 0; t < 24; t++) {
            int idx = pi_x[t];
            uint64_t next = state[idx];
            B[idx] = rotl64(cur, ro[t]);
            cur = next;
        }

        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 5; x++) {
                state[y*5+x] =
                    B[y*5+x] ^
                    ((~B[y*5+(x+1)%5]) & B[y*5+(x+2)%5]);
            }
        }

        state[0] ^= RC[r];
    }
}

__device__ void keccak256(
    const uint8_t *input,
    uint32_t inlen,
    uint8_t output[32]
) {
    const int RATE = 136;

    uint8_t buf[RATE];
    uint64_t state[25];

    memset(state, 0, sizeof(state));
    memset(buf, 0, RATE);

    for (uint32_t i = 0; i < inlen; i++)
        buf[i] = input[i];

    buf[inlen] = 0x01;
    buf[RATE - 1] |= 0x80;

    for (int i = 0; i < RATE / 8; i++) {
        uint64_t lane = 0;

        for (int j = 0; j < 8; j++)
            lane |= ((uint64_t)buf[i*8+j]) << (j*8);

        state[i] ^= lane;
    }

    keccak_f(state);

    for (int i = 0; i < 4; i++) {
        uint64_t lane = state[i];

        for (int j = 0; j < 8; j++)
            output[i*8+j] = (lane >> (j*8)) & 0xFF;
    }
}

extern "C" __global__ void mine_kernel(
    const uint32_t *challenge_be8,
    const uint32_t *difficulty_be8,
    uint64_t        base_nonce,
    uint64_t        batch_size,
    uint32_t       *result
) {
    uint64_t gid =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= batch_size)
        return;

    if (result[0])
        return;

    uint64_t nonce = base_nonce + gid;

    uint8_t input[64];

    for (int i = 0; i < 8; i++) {
        uint32_t w = challenge_be8[i];

        input[i*4+0] = (w >> 24) & 0xFF;
        input[i*4+1] = (w >> 16) & 0xFF;
        input[i*4+2] = (w >>  8) & 0xFF;
        input[i*4+3] =  w        & 0xFF;
    }

    for (int i = 32; i < 56; i++)
        input[i] = 0;

    input[56] = (nonce >> 56) & 0xFF;
    input[57] = (nonce >> 48) & 0xFF;
    input[58] = (nonce >> 40) & 0xFF;
    input[59] = (nonce >> 32) & 0xFF;
    input[60] = (nonce >> 24) & 0xFF;
    input[61] = (nonce >> 16) & 0xFF;
    input[62] = (nonce >>  8) & 0xFF;
    input[63] =  nonce        & 0xFF;

    uint8_t hash[32];

    keccak256(input, 64, hash);

    bool below = true;

    for (int i = 0; i < 8; i++) {
        uint32_t hw =
            ((uint32_t)hash[i*4+0] << 24) |
            ((uint32_t)hash[i*4+1] << 16) |
            ((uint32_t)hash[i*4+2] <<  8) |
             (uint32_t)hash[i*4+3];

        uint32_t dw = difficulty_be8[i];

        if (hw < dw) {
            below = true;
            break;
        }

        if (hw > dw) {
            below = false;
            break;
        }
    }

    if (below) {
        if (atomicCAS(&result[0], 0u, 1u) == 0) {
            result[1] = (uint32_t)(nonce & 0xFFFFFFFFULL);
            result[2] = (uint32_t)(nonce >> 32);
        }
    }
}