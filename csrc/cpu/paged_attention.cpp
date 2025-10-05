#include <ATen/TensorIndexing.h>
#include <ATen/core/jit_type.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/logging_is_not_google_glog.h>

#include <cstdint>
#include <iostream>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

// [NOTE] TODO list for this kernel:
//   1. tune the value for BLOCK_N
//   2. planning for {batches, num_heads, num_kv_splits}
//      and use actual num_kv_splits for small seq length
//   3. try fast impl of `.tanh()`
//   4. provide amx kernel for index_gemm_kernel_nn when M = 16
//
template <typename index_t>
inline index_t get_index(index_t* ind, int i) {
  return (ind == nullptr) ? (index_t)i : ind[i];
}

#if defined(CPU_CAPABILITY_AVX512)
// key: from [N, 32] to [32/2, N, 2]
template <typename scalar_t, typename index_t>
inline void pack_vnni_Nx32(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int N,
    int ld_src,
    int ld_dst) {
  __m512i vinputs[16];

  int n = 0;
  for (; n < N; ++n) {
    index_t index = get_index(ind, n);
    vinputs[n] = _mm512_loadu_si512(src + index * ld_src);
  }
  // padding with zero to avoid uninitialized vectors
  for (; n < 16; ++n) {
    vinputs[n] = _mm512_set1_epi32(0);
  }

  // pack key
  transpose_16x16_32bit(vinputs);

  const __mmask16 vmask = (1 << N) - 1;
  for (int k = 0; k < 16; ++k) {
    _mm512_mask_storeu_epi32(dst + k * ld_dst * 2, vmask, vinputs[k]);
  }
}

// value: from [K, 32] to [K/2, 32, 2]
template <typename scalar_t, typename index_t>
inline void pack_vnni_Kx32(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int K,
    int ld_src,
    int ld_dst) {
  __m512i vinputs[2];

  int k = 0;
  for (; k < K; ++k) {
    index_t index = get_index(ind, k);
    vinputs[k] = _mm512_loadu_si512(src + index * ld_src);
  }
  // padding with zero to avoid uninitialized vectors
  for (; k < 2; ++k) {
    vinputs[k] = _mm512_set1_epi32(0);
  }

  // pack value
  __m512i d0, d1;
  std::tie(d0, d1) = transpose_2x32_16bit(vinputs[0], vinputs[1]);
  _mm512_storeu_si512(dst + 0 * ld_dst * 2, d0);
  _mm512_storeu_si512(dst + 0 * ld_dst * 2 + 32, d1);
}
#endif

// convert to vnni format
// from [N, K/2, 2] to [K/2, N, 2] for bfloat16 and float16
template <typename scalar_t, typename index_t>
void pack_vnni(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int N,
    int K,
    int ld_src,
    int ld_dst) {
#if defined(CPU_CAPABILITY_AVX512)
  const int NB = div_up(N, 16);
  const int KB = K / 32;  // no remainder
  const bool is_indexed = ind != nullptr;

  for (int nb = 0; nb < NB; ++nb) {
    for (int kb = 0; kb < KB; ++kb) {
      // handle 16x512bits each block
      int nb_size = std::min(N - nb * 16, 16);
      pack_vnni_Nx32<scalar_t, index_t>(
          /*    dst */ dst + ((kb * 32) >> 1) * ld_dst * 2 + nb * 16 * 2,
          /*    src */ src + kb * 32 + (is_indexed ? 0 : nb * 16 * ld_src),
          /*    ind */ is_indexed ? ind + nb * 16 : nullptr,
          /*      N */ nb_size,
          /* ld_src */ ld_src,
          /* ld_dst */ ld_dst);
    }
  }
#else
  for (int n = 0; n < N; ++n) {
    index_t index = get_index(ind, n);
    for (int k = 0; k < K / 2; ++k) {
      for (int d = 0; d < 2; ++d) {
        dst[k * ld_dst * 2 + n * 2 + d] = src[index * ld_src + k * 2 + d];
      }
    }
  }
#endif
}

// convert to vnni format
// from [K/2, 2, N] to [K/2, N, 2] for bfloat16 and float16
template <typename scalar_t, typename index_t>
void pack_vnni2(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int K,
    int N,
    int ld_src,
    int ld_dst) {
#if defined(CPU_CAPABILITY_AVX512)
  const int KB = div_up(K, 2);
  const int NB = N / 32;  // no remainder
  const bool is_indexed = ind != nullptr;

  for (int kb = 0; kb < KB; ++kb) {
    for (int nb = 0; nb < NB; ++nb) {
      // handle 2x512bits each block
      int kb_size = std::min(K - kb * 2, 2);
      pack_vnni_Kx32<scalar_t, index_t>(
          /*    dst */ dst + ((kb * 2) >> 1) * ld_dst * 2 + nb * 32 * 2,
          /*    src */ src + (is_indexed ? 0 : kb * 2 * ld_src) + nb * 32,
          /*    ind */ is_indexed ? ind + kb * 2 : nullptr,
          /*      K */ kb_size,
          /* ld_src */ ld_src,
          /* ld_dst */ ld_dst);
    }
  }
#else
  int k = 0;
  for (; k < (K >> 1) * 2; k += 2) {
    index_t index0 = get_index(ind, k + 0);
    index_t index1 = get_index(ind, k + 1);
    for (int n = 0; n < N; ++n) {
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 0] = src[index0 * ld_src + n];
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 1] = src[index1 * ld_src + n];
    }
  }
  if (K % 2 != 0) {
    index_t index = get_index(ind, K - 1);
    for (int n = 0; n < N; ++n) {
      dst[(K >> 1) * ld_dst * 2 + n * 2 + 0] = src[index * ld_src + n];
      dst[(K >> 1) * ld_dst * 2 + n * 2 + 1] = 0;
    }
    k += 2;
  }
#endif
}

#if defined(CPU_CAPABILITY_AVX512)
// key: from [N, 32] to [32/2, N, 2]
// val: from [N, 32] to [N/2, 32, 2]
template <typename scalar_t, typename index_t>
inline void pack_vnni_Nx32(
    scalar_t* __restrict__ dst0,
    scalar_t* __restrict__ dst1,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int N,
    int ld_src,
    int ld_dst0,
    int ld_dst1,
    bool convert_v) {
  __m512i vinputs[16];
  int n = 0;
  for (; n < N; ++n) {
    vinputs[n] = _mm512_loadu_si512(src + ind[n] * ld_src);
  }
  // padding with zero to avoid uninitialized vectors
  for (; n < 16; ++n) {
    vinputs[n] = _mm512_set1_epi32(0);
  }

  // pack value, skip 64 elems for deepseek
  // handle 2 vectors at a time from [2, 32] to [32, 2]
  if (convert_v) {
    for (int n = 0; n < 16; n += 2) {
      __m512i d0, d1;
      std::tie(d0, d1) = transpose_2x32_16bit(vinputs[n], vinputs[n + 1]);
      _mm512_storeu_si512(dst1 + (n >> 1) * ld_dst1 * 2, d0);
      _mm512_storeu_si512(dst1 + (n >> 1) * ld_dst1 * 2 + 32, d1);
    }
  }

  // pack key
  transpose_16x16_32bit(vinputs);

  const __mmask16 vmask = (1 << N) - 1;
  for (int k = 0; k < 16; ++k) {
    _mm512_mask_storeu_epi32(dst0 + k * ld_dst0 * 2, vmask, vinputs[k]);
  }
}
#endif

// [NOTE]: MLA vnni format conversion
//
//  here we apply same strategy as `FlashMLA`:
//  each kv_cache is loaded once and packed twice (L2 cache hit)
//
//  * for   key: from [N, K/2, 2] to [K/2, N, 2]
//  * for value: from [N/2, 2, Kv] to [N/2, Kv, 2]
//
template <typename scalar_t, typename index_t>
void pack_vnni(
    scalar_t* __restrict__ dst0,
    scalar_t* __restrict__ dst1,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int N,
    int K,
    int Kv,
    int ld_src,
    int ld_dst0,
    int ld_dst1) {
#if defined(CPU_CAPABILITY_AVX512)
  const int NB = div_up(N, 16);
  const int KB = K / 32;    // no remainder
  const int KBv = Kv / 32;  // no remainder

  for (int nb = 0; nb < NB; ++nb) {
    for (int kb = 0; kb < KB; ++kb) {
      // handle 16x512bits each block
      int nb_size = std::min(N - nb * 16, 16);
      pack_vnni_Nx32<scalar_t, index_t>(
          /*    dst0 */ dst0 + ((kb * 32) >> 1) * ld_dst0 * 2 + nb * 16 * 2,
          /*    dst1 */ dst1 + ((nb * 16) >> 1) * ld_dst1 * 2 + kb * 32 * 2,
          /*     src */ src + kb * 32,
          /*     ind */ ind + nb * 16,
          /*       N */ nb_size,
          /*  ld_src */ ld_src,
          /* ld_dst0 */ ld_dst0,
          /* ld_dst1 */ ld_dst1,
          /*   cvt_v */ kb < KBv);
    }
  }
#else
  for (int n = 0; n < N; ++n) {
    index_t index = ind[n];
    for (int k = 0; k < K / 2; ++k) {
      for (int d = 0; d < 2; ++d) {
        dst0[k * ld_dst0 * 2 + n * 2 + d] = src[index * ld_src + k * 2 + d];
      }
    }
  }
  // from [N/2, 2, K] to [N/2, K, 2]
  for (int n = 0; n < (N >> 1) * 2; n += 2) {
    index_t index0 = ind[n + 0];
    index_t index1 = ind[n + 1];
    for (int k = 0; k < Kv; ++k) {
      dst1[(n >> 1) * ld_dst1 * 2 + k * 2 + 0] = src[index0 * ld_src + k];
      dst1[(n >> 1) * ld_dst1 * 2 + k * 2 + 1] = src[index1 * ld_src + k];
    }
  }
  if (N % 2 != 0) {
    index_t index = ind[N - 1];
    for (int k = 0; k < Kv; ++k) {
      dst1[(N >> 1) * ld_dst1 * 2 + k * 2 + 0] = src[index * ld_src + k];
      dst1[(N >> 1) * ld_dst1 * 2 + k * 2 + 1] = 0;
    }
  }
#endif
}

template <typename scalar_t>
inline void fill_stub(scalar_t* __restrict__ out, float val, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = Vec::size();
  const Vec data_vec = Vec(static_cast<scalar_t>(val));
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    data_vec.store(out + d);
  }
  if (size - d > 0) {
    data_vec.store(out + d, size - d);
  }
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ acc, float s, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec s_fvec = fVec(s);
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    fVec a_fvec0 = fVec::loadu(acc + d) * s_fvec;
    fVec a_fvec1 = fVec::loadu(acc + d + fVec::size()) * s_fvec;
    bVec out_bvec = convert_from_float_ext<scalar_t>(a_fvec0, a_fvec1);
    out_bvec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(acc[d] * s);
  }
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ src, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = bVec::size();
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    bVec out_bvec = bVec::loadu(src + d);
    out_bvec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = src[d];
  }
}

template <typename scalar_t, int BLOCK_N>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input) {
  static_assert(BLOCK_N % 32 == 0);
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int COLS = BLOCK_N / 16;
  auto store = [&](auto i) {
    constexpr int col = i % COLS;
    // for COLS = 2, 4 use 512bit store
    if constexpr (col % 2 == 0) {
      fVec a_fvec0 = fVec::loadu(input + col * 16);
      fVec a_fvec1 = fVec::loadu(input + col * 16 + 16);
      bVec out_bvec = convert_from_float_ext<scalar_t>(a_fvec0, a_fvec1);
      out_bvec.store(out + col * 16);
    }
  };
  Unroll<COLS>{}(store);
}

// GEMM handles query @ key (indexed) x scale
//   A : [M, K]
//   B : [N, K] indexed
//   C : [M, N]
//
template <typename scalar_t, typename index_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nt {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      float scale,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t K,
      int64_t max_tokens) {
    for (int64_t m = 0; m < BLOCK_M; ++m) {
      for (int64_t n = 0; n < BLOCK_N; ++n) {
        float sum = 0.f;
        int64_t b_idx = indices[n];
        TORCH_CHECK(b_idx < max_tokens, "token index out of scope!");
        for (int64_t k = 0; k < K; ++k) {
          sum += scale * static_cast<float>(A[m * lda + k]) * static_cast<float>(B[b_idx * ldb + k]);
        }
        C[m * ldc + n] = sum;
      }
    }
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <typename index_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nt<at::BFloat16, index_t, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::BFloat16* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      float scale,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t K,
      int64_t max_tokens) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];
    __m512 vscale = _mm512_set1_ps(scale);

    auto loadc = [&](auto i) { vc[i] = _mm512_setzero_ps(); };
    Unroll<ROWS * COLS>{}(loadc);

    // for main loop
    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_loadu_si512(A + row * lda + k));
      }
      if constexpr (row == 0) {
        if constexpr (col + 1 < COLS) {
          int64_t b_idx_prefetch = indices[col + 1];
          _mm_prefetch(B + b_idx_prefetch * ldb + k, _MM_HINT_T0);
        }
        int64_t b_idx = indices[col];
        TORCH_CHECK(b_idx < max_tokens, "token index out of scope!");
        vb[col] = (__m512bh)(_mm512_loadu_si512(B + b_idx * ldb + k));
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };

    // for remainder
    auto compute2 = [&](auto i, int64_t k, __mmask32 mask) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_maskz_loadu_epi16(mask, A + row * lda + k));
      }
      if constexpr (row == 0) {
        int64_t b_idx = indices[col];
        TORCH_CHECK(b_idx < max_tokens, "token index out of scope!");
        vb[col] = (__m512bh)(_mm512_maskz_loadu_epi16(mask, B + b_idx * ldb + k));
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };

    int64_t k = 0;
    for (; k <= K - 32; k += 32) {
      Unroll<ROWS * COLS>{}(compute, k);
    }
    int64_t count = K - k;
    if (count > 0) {
      __mmask32 mask = (1ULL << count) - 1;
      Unroll<ROWS * COLS>{}(compute2, k, mask);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      C[row * ldc + col] = _mm512_reduce_add_ps(_mm512_mul_ps(vc[i], vscale));
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NT(MB_SIZE, NB_SIZE)               \
  tinygemm_kernel_nt<scalar_t, index_t, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda, B, C + mb_start * ldc + nb_start, indices + nb_start, scale, lda, ldb, ldc, K, max_tokens);

// this is used when N isn't multiple of 16,
// N corresponds to `head_size_v` which should be 16x
template <typename scalar_t, typename index_t>
inline void tinygemm_kernel_nn_scalar(
    const float* __restrict__ A,
    const scalar_t* __restrict__ B,
    float* __restrict__ C,
    const index_t* __restrict__ indices,
    const float* __restrict__ scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t max_tokens) {
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      C[m * ldc + n] *= scale[m];
      for (int64_t k = 0; k < K; ++k) {
        int64_t b_idx = indices[k];
        TORCH_CHECK(b_idx < max_tokens, "token index out of scope!");
        C[m * ldc + n] += A[m * lda + k] * static_cast<float>(B[b_idx * ldb + n]);
      }
    }
  }
}

// GEMM handles v' * scale + attn @ value (indexed)
//   A : [M, K]
//   B : [K, N] indexed
//   C ：[M, N]
//
template <typename scalar_t, typename index_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const float* __restrict__ A,
      const scalar_t* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      const float* __restrict__ scale,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t K,
      int64_t max_tokens) {
    tinygemm_kernel_nn_scalar(A, B, C, indices, scale, BLOCK_M, BLOCK_N, K, lda, ldb, ldc, max_tokens);
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <typename index_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, index_t, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const float* __restrict__ A,
      const at::BFloat16* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      const float* __restrict__ scale,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t K,
      int64_t max_tokens) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    __m512 va;
    __m512 vb[COLS];
    __m512 vc[ROWS * COLS];
    __m512 vscale;

    auto loadc = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
      if constexpr (col == 0) {
        vscale = _mm512_set1_ps(scale[row]);
      }
#pragma GCC diagnostic pop
      vc[i] = _mm512_loadu_ps(C + row * ldc + col * 16);
      vc[i] = _mm512_mul_ps(vc[i], vscale);
    };
    Unroll<ROWS * COLS>{}(loadc);

    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = _mm512_set1_ps(A[row * lda + k]);
      }
      if constexpr (row == 0) {
        if (k + 1 < K) {
          int64_t b_idx_prefetch = indices[k + 1];
          _mm_prefetch(B + b_idx_prefetch * ldb + col * 16, _MM_HINT_T0);
        }
        int64_t b_idx = indices[k];
        TORCH_CHECK(b_idx < max_tokens, "token index out of scope!");

        // for COLS = 2, 4, 6, 8 use 512 bit load
        // for COLS = 1, 3, 5, 7 use 256 bit load
        if constexpr (COLS % 2 == 0) {
          if constexpr (col % 2 == 0) {
            __m512i b16 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(B + b_idx * ldb + col * 16));
            vb[col + 0] = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(b16, 0));
            vb[col + 1] = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(b16, 1));
          }
        } else {
          __m256i b16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B + b_idx * ldb + col * 16));
          vb[col] = CVT_BF16_TO_FP32(b16);
        }
      }
      vc[i] = _mm512_fmadd_ps(va, vb[col], vc[i]);
    };

    for (int64_t k = 0; k < K; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      _mm512_storeu_ps(C + row * ldc + col * 16, vc[i]);
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)               \
  tinygemm_kernel_nn<scalar_t, index_t, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda,                                         \
      B + nb_start,                                               \
      C + mb_start * ldc + nb_start,                              \
      indices,                                                    \
      scale + mb_start,                                           \
      lda,                                                        \
      ldb,                                                        \
      ldc,                                                        \
      K,                                                          \
      max_tokens);

template <typename scalar_t, typename index_t>
void index_gemm_kernel_nt(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    float* __restrict__ C,
    const index_t* __restrict__ indices,
    float scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t max_tokens) {
  // pattern: 1-8-8
  if (M == 1) {
    constexpr int64_t BLOCK_N = 8;
    const int64_t NB = div_up(N, BLOCK_N);
    int64_t mb_start = 0, lda = 1, ldc = 1;

    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (nb_size) {
        case 1:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 1);
          break;
        case 2:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 2);
          break;
        case 3:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 3);
          break;
        case 4:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 4);
          break;
        case 5:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 5);
          break;
        case 6:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 6);
          break;
        case 7:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 7);
          break;
        case 8:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 8);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, 1x", "nb_size");
      }
    }
    return;
  }

  // pattern: 1-6-24
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 6;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  for (int64_t mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size) {
        // mb_size = 1
        case 0x11:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 1);
          break;
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 2);
          break;
        case 0x13:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 3);
          break;
        case 0x14:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 4);
          break;
        case 0x15:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 5);
          break;
        case 0x16:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 6);
          break;
        // mb_size = 2
        case 0x21:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 1);
          break;
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 2);
          break;
        case 0x23:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 3);
          break;
        case 0x24:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 4);
          break;
        case 0x25:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 5);
          break;
        case 0x26:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 6);
          break;
        // mb_size = 3
        case 0x31:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 1);
          break;
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 2);
          break;
        case 0x33:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 3);
          break;
        case 0x34:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 4);
          break;
        case 0x35:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 5);
          break;
        case 0x36:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 6);
          break;
        // mb_size = 4
        case 0x41:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 1);
          break;
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 2);
          break;
        case 0x43:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 3);
          break;
        case 0x44:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 4);
          break;
        case 0x45:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 5);
          break;
        case 0x46:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 6);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

template <typename scalar_t, typename index_t>
void index_gemm_kernel_nn(
    const float* __restrict__ A,
    const scalar_t* __restrict__ B,
    float* __restrict__ C,
    const index_t* __restrict__ indices,
    float* __restrict__ scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t max_tokens) {
  constexpr int kVecSize = 16;
  if ((N & (kVecSize - 1)) != 0) {
    tinygemm_kernel_nn_scalar(A, B, C, indices, scale, M, N, K, lda, ldb, ldc, max_tokens);
    return;
  }

  // pattern: 1-8-8
  if (M == 1) {
    constexpr int64_t BLOCK_N = 8 * kVecSize;
    const int64_t NB = div_up(N, BLOCK_N);
    int64_t mb_start = 0, lda = 1, ldc = 1;

    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (nb_size >> 4) {
        case 1:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 16);
          break;
        case 2:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 32);
          break;
        case 3:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 48);
          break;
        case 4:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 64);
          break;
        case 5:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 80);
          break;
        case 6:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 96);
          break;
        case 7:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 112);
          break;
        case 8:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 128);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, 1x", "nb_size");
      }
    }
    return;
  }

  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 6 * kVecSize;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  for (int64_t mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size >> 4) {
        // mb_size = 1
        case 0x11:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 16);
          break;
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 32);
          break;
        case 0x13:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 48);
          break;
        case 0x14:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 64);
          break;
        case 0x15:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 80);
          break;
        case 0x16:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 96);
          break;
        // mb_size = 2
        case 0x21:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 16);
          break;
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 32);
          break;
        case 0x23:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 48);
          break;
        case 0x24:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 64);
          break;
        case 0x25:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 80);
          break;
        case 0x26:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 96);
          break;
        // mb_size = 3
        case 0x31:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 16);
          break;
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 32);
          break;
        case 0x33:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 48);
          break;
        case 0x34:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 64);
          break;
        case 0x35:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 80);
          break;
        case 0x36:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 96);
          break;
        // mb_size = 4
        case 0x41:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 16);
          break;
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 32);
          break;
        case 0x43:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 48);
          break;
        case 0x44:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 64);
          break;
        case 0x45:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 80);
          break;
        case 0x46:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 96);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

template <typename scalar_t>
void decode_accumulate_kv_splits(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const int32_t* __restrict__ req_pool_indices,
    const int32_t* __restrict__ cu_q_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t l_stride0,
    int64_t l_stride1,
    int64_t l_stride2) {
  using Vec = at::vec::Vectorized<float>;

  // parallel on [batches, num_heads]
  at::parallel_for(0, batches * num_heads, 0, [&](int64_t begin, int64_t end) {
    // NB: here we use logits[b][h][0] as acc, since
    // for the first kv split (kv_id == 0):
    //   m_delta = std::exp(-inf) = 0
    //   e_logic = std::exp(0) = 1
    //   acc = acc * m_delta + tv * e_logic = tv
    for (int64_t i = begin; i < end; ++i) {
      int64_t bs{0}, head_id{0};
      data_index_init(i, bs, batches, head_id, num_heads);
      auto global_bs = req_pool_indices[bs];

      float* __restrict__ acc = attn_logits + bs * l_stride0 + head_id * l_stride1;

      float s_prime = 0.f;
      float m_prime = -std::numeric_limits<float>::infinity();

      // update acc with from each kv_split
      for (int64_t kv_id = 0; kv_id < num_kv_splits; ++kv_id) {
        float* __restrict__ tv = acc + kv_id * l_stride2;
        const float tlogic = (acc + kv_id * l_stride2)[head_size_v];

        float m_i = std::max(tlogic, m_prime);
        float m_delta = std::exp(m_prime - m_i);
        float e_logic = std::exp(tlogic - m_i);
        if (kv_id != 0) {
          at::vec::map2<float>(
              [m_delta, e_logic](Vec x, Vec y) { return x * Vec(m_delta) + y * Vec(e_logic); },
              acc,
              acc,
              tv,
              head_size_v);
        }

        s_prime = s_prime * m_delta + e_logic;
        m_prime = m_i;
      }

      copy_stub<scalar_t>(
          output + (cu_q_lens[global_bs] * num_heads + head_id) * head_size_v, acc, 1 / s_prime, head_size_v);
    }
  });
}

template <typename scalar_t, typename index_t, int64_t BLOCK_N>
void decode_paged_attention_grouped_kernel_impl(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ k_buffer,
    const scalar_t* __restrict__ v_buffer,
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    float* __restrict__ lse,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ req_pool_indices,
    const int32_t* __restrict__ cu_q_lens,
    const int32_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t block_size,
    int64_t max_block_cnt,
    int64_t max_total_num_tokens) {
  using Vec = at::vec::Vectorized<float>;

  // block length for heads
  // we parallel on [batches, divup(num_heads, BLOCK_H), num_kv_splits]
  // use smaller BLOCK_H when batches is small to utilize all cores
  constexpr int64_t kBLOCK_H = 16;
  const int64_t BLOCK_H = std::min(4 * batches, kBLOCK_H);

  // strides
  const int64_t l_stride0 = num_heads * num_kv_splits * (head_size_v + 1);
  const int64_t l_stride1 = num_kv_splits * (head_size_v + 1);
  const int64_t l_stride2 = head_size_v + 1;

  const bool has_logit_cap = logit_cap > 0;
  float rlogit_cap = has_logit_cap ? 1 / logit_cap : 0.f;

  // partition the heads into blocks for parallel
  const int64_t num_groups = num_heads / num_heads_kv;
  const int64_t num_blocks = div_up(num_groups, BLOCK_H);

  // parallel on [batches, num_heads_kv, num_blocks, num_kv_splits]
  at::parallel_for(0, batches * num_heads_kv * num_blocks * num_kv_splits, 0, [&](int64_t begin, int64_t end) {
    // for (int64_t idx = 0; idx < batches * num_heads_kv * num_blocks * num_kv_splits; idx++) {
    // int64_t begin = idx;
    // int64_t end = begin + 1;

    int64_t bs{0}, head_kv_id{0}, block_id{0}, kv_id{0};
    data_index_init(begin, bs, batches, head_kv_id, num_heads_kv, block_id, num_blocks, kv_id, num_kv_splits);

    alignas(64) float s_i[BLOCK_H * BLOCK_N];
    float* __restrict__ s_delta = s_i;

    alignas(64) float s_prime[BLOCK_H];
    alignas(64) float m_prime[BLOCK_H];
    alignas(64) float m_delta[BLOCK_H];

    for (int64_t i = begin; i < end; ++i) {
      const int64_t h_start = head_kv_id * num_groups + block_id * BLOCK_H;
      const int64_t h_end = head_kv_id * num_groups + std::min(block_id * BLOCK_H + BLOCK_H, num_groups);
      const int64_t h_size = h_end - h_start;

      int64_t global_bs = req_pool_indices[bs];
      int64_t seq_len_kv = seq_lens[global_bs];

      // get query
      const scalar_t* __restrict__ q_ptr = query + cu_q_lens[global_bs] * q_strideM + h_start * q_strideH;
      const int32_t* __restrict__ req_block_table = block_table + global_bs * max_block_cnt;

      const int64_t SPLIT_SIZE = div_up(seq_len_kv, num_kv_splits);
      const int64_t kv_start = kv_id * SPLIT_SIZE;
      const int64_t kv_end = std::min(kv_start + SPLIT_SIZE, seq_len_kv);

      fill_stub(s_prime, 0.f, BLOCK_H);
      fill_stub(m_prime, -std::numeric_limits<float>::infinity(), BLOCK_H);

      // get v_prime, and init to zero
      float* __restrict__ v_prime = attn_logits + bs * l_stride0 + h_start * l_stride1 + kv_id * l_stride2;
      for (int64_t h = 0; h < h_size; ++h) {
        fill_stub(v_prime + h * l_stride1, 0.f, head_size_v);
      }

      // loop over K and V sequence with BLOCK_N
      for (int64_t n = kv_start; n < kv_end; n += BLOCK_N) {
        int64_t n_size = std::min(BLOCK_N, kv_end - n);

        alignas(64) index_t req_to_token[BLOCK_N];
        for (int i = 0; i < n_size; i++) {
          req_to_token[i] = req_block_table[(n + i) / block_size] * block_size + (n + i) % block_size;
        }

        // calculate Q @ K
        index_gemm_kernel_nt<scalar_t, index_t>(
            /* A   */ q_ptr,
            /* B   */ k_buffer + head_kv_id * k_strideH,
            /* C   */ s_i,
            /* ind */ req_to_token,
            /* scl */ scaling,
            /* M   */ h_size,
            /* N   */ n_size,
            /* K   */ head_size,
            /* lda */ q_strideH,
            /* ldb */ k_strideN,
            /* ldc */ BLOCK_N,
            /* mtt */ max_total_num_tokens);

        if (has_logit_cap) {
          at::vec::map<float>(
              [logit_cap, rlogit_cap](Vec x) { return Vec(logit_cap) * (x * Vec(rlogit_cap)).tanh(); },
              s_i,
              s_i,
              BLOCK_H * BLOCK_N);
        }

        // update the scaling coefficients
        for (int64_t h = 0; h < h_size; ++h) {
          // m_i: max value per row
          float m_i = at::vec::reduce_all<float>(
              [](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, s_i + h * BLOCK_N, n_size);
          m_i = std::max(m_i, m_prime[h]);

          // m_delta <- exp(m' - m_i)
          m_delta[h] = std::exp(m_prime[h] - m_i);

          // s_delta <- exp(s_i - m_i)
          at::vec::map<float>(
              [m_i](Vec x) { return (x - Vec(m_i)).exp_u20(); }, s_delta + h * BLOCK_N, s_i + h * BLOCK_N, n_size);

          // s' <- s' * m_delta + sum(s_delta)
          s_prime[h] *= m_delta[h];
          s_prime[h] += at::vec::reduce_all<float>([](Vec& x, Vec& y) { return x + y; }, s_delta + h * BLOCK_N, n_size);

          m_prime[h] = m_i;
        }

        // calculate V' <- s_delta @ V + V' * m_delta
        index_gemm_kernel_nn<scalar_t, index_t>(
            /* A   */ s_delta,
            /* B   */ v_buffer + head_kv_id * v_strideH,
            /* C   */ v_prime,
            /* ind */ req_to_token,
            /* scl */ m_delta,
            /* M   */ h_size,
            /* N   */ head_size_v,
            /* K   */ n_size,
            /* lda */ BLOCK_N,
            /* ldb */ v_strideN,
            /* ldc */ l_stride1,
            /* mtt */ max_total_num_tokens);
      }  // loop with KV blocks

      // only update v' when kv_split_size > 0; otherwise mark tlogic as -inf
      if (kv_end > kv_start) {
        for (int64_t h = 0; h < h_size; ++h) {
          float s = 1 / s_prime[h];
          at::vec::map<float>(
              [s](Vec out) { return out * Vec(s); }, v_prime + h * l_stride1, v_prime + h * l_stride1, head_size_v);
          (v_prime + h * l_stride1)[head_size_v] = m_prime[h] + std::log(s_prime[h]);
        }
      } else {
        for (int64_t h = 0; h < h_size; ++h) {
          (v_prime + h * l_stride1)[head_size_v] = -std::numeric_limits<float>::infinity();
        }
      }

      // move to the next index
      data_index_step(bs, batches, head_kv_id, num_heads_kv, block_id, num_blocks, kv_id, num_kv_splits);
    }
  });

  decode_accumulate_kv_splits(
      output,
      attn_logits,
      req_pool_indices,
      cu_q_lens,
      batches,
      num_heads,
      head_size_v,
      num_kv_splits,
      l_stride0,
      l_stride1,
      l_stride2);
  // compute the LSE in last stage
  // lse = attn_logits.index({at::indexing::Slice(), at::indexing::Slice(), at::indexing::Slice(),
  // -1}).logsumexp(-1).squeeze()
  at::parallel_for(0, batches * num_heads, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      int64_t bs{0}, head_id{0};
      data_index_init(i, bs, batches, head_id, num_heads);
      auto global_bs = req_pool_indices[bs];

      float* __restrict__ acc = attn_logits + bs * l_stride0 + head_id * l_stride1;

      float m_i = -std::numeric_limits<float>::infinity();
      for (int64_t kv_id = 0; kv_id < num_kv_splits; ++kv_id) {
        // use Vec reduction to find max
        float tlogic = (acc + kv_id * l_stride2)[head_size_v];
        m_i = std::max(tlogic, m_i);
      }

      float lse_i = 0.f;
      for (int64_t kv_id = 0; kv_id < num_kv_splits; ++kv_id) {
        float tlogic = (acc + kv_id * l_stride2)[head_size_v];
        lse_i += std::exp(tlogic - m_i);
      }
      lse[cu_q_lens[global_bs] * num_heads + head_id] = m_i + std::log(lse_i);
    }
  });

}  // GQA/MQA

/*
  Prefill paged attention with grouped QKV
  Similar to prefill_paged_attention_kernel_impl, but with grouped QKV

  output:
    o: [output_len, num_heads, head_size_v] // must contiguous
    logsumexp_extend: [output_len, batches, num_heads]
    q: [output_len, num_heads, head_size] // must contiguous
    k_buffer: [block_cnt, block_size, kv_head_dim, head_size]
    v_buffer: [block_cnt, block_size, kv_head_dim, head_size_v]
    block_table: [batches, max_context_len] (in blocks)
    req_pool_indices: [batches] (in requests)
    seq_lens: [batches] (in tokens)
    q_lens: [batches] (in tokens)
    cu_q_lens: [batches] (in tokens)
    buffer: per-thread buffer
  args:
    batches: number of requests in the batch
    num_heads: number of attention heads
    num_heads_kv: number of attention heads for K and V
    head_size: head dimension for Q and K
    head_size_v: head dimension for V
    q_strideM: stride between two rows of Q
    q_strideH: stride between two heads of Q
    k_strideN: stride between two rows of K
    k_strideH: stride between two heads of K
    v_strideN: stride between two rows of V
    v_strideH: stride between two heads of V
    scaling: scaling factor for QK^T
    logit_cap: cap for attention logits (0 means no cap)
    max_block_cnt: maximum context length
    max_extend_len: maximum extension length (in tokens)
    buffer_size_per_thread: size of the per-thread buffer (in bytes)
  template args:
    scalar_t: data type for Q, K, V, and output
*/
template <typename scalar_t, typename index_t, int BLOCK_M, int BLOCK_N>
void prefill_paged_attention_grouped_kernel_impl(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_buffer,
    const scalar_t* __restrict__ v_buffer,
    scalar_t* __restrict__ output,
    float* __restrict__ lse_output,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ req_pool_indices,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ q_lens,
    const int32_t* __restrict__ cu_q_lens,
    const void* __restrict__ buffer,
    int batches,
    int num_heads,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int q_strideM,
    int q_strideH,
    int k_strideN,
    int k_strideH,
    int v_strideN,
    int v_strideH,
    float scaling,
    float logit_cap,
    const int max_num_reqs,
    const int max_block_cnt,
    const int max_extend_len,
    const int block_size,
    int buffer_size_per_thread) {
  using Vec = at::vec::Vectorized<float>;

  // strides
  const int o_strideM = num_heads * head_size_v;
  const int o_strideH = head_size_v;

  // we use same buffer for packed key and value
  const int ldb_tmp = std::max(head_size, head_size_v);

  const bool has_logit_cap = logit_cap > 0;
  float rlogit_cap = has_logit_cap ? 1 / logit_cap : 0.f;

  const int num_groups = num_heads / num_heads_kv;
  TORCH_CHECK(num_groups * num_heads_kv == num_heads);

  // number of blocks along M
  int MB = div_up(max_extend_len, BLOCK_M);

  // parallel on [batches, num_heads, BM]
  at::parallel_for(0, batches * num_heads * MB, 0, [&](int begin, int end) {
    // for (int z = 0; z < batches * num_heads * MB; z++) {
    // int begin = z;
    // int end = begin + 1;
    int batch{0}, head_id{0}, mb{0};
    data_index_init(begin, batch, batches, head_id, num_heads, mb, MB);

    int tid = at::get_thread_num();
    // s_i and s_delta: [BLOCK_M, BLOCK_N]
    float* __restrict__ s_i = reinterpret_cast<float*>((char*)(buffer) + tid * buffer_size_per_thread);
    float* __restrict__ s_delta = s_i;

    // v_prime: [BLOCK_M, head_size_v]
    float* __restrict__ v_prime = s_i + BLOCK_M * BLOCK_N;

    // s_delta2: [BLOCK_M, BLOCK_N]; copy of s_delta in scalar_t
    scalar_t* __restrict__ s_delta2 = reinterpret_cast<scalar_t*>(v_prime + BLOCK_N * head_size_v);

    // Btmp: [BLOCK_N, max(head_size, head_size_v)]
    scalar_t* __restrict__ Btmp = s_delta2 + BLOCK_M * BLOCK_N;

    // init Btmp just once for each thread to prevent NaN
    fill_stub(Btmp, 0.f, BLOCK_N * ldb_tmp);

    alignas(64) thread_local float s_prime[BLOCK_M];
    alignas(64) thread_local float m_prime[BLOCK_M];

    for (int i = begin; i < end; ++i) {
      int head_kv_id = head_id / num_groups;
      int req_pool_id = req_pool_indices[batch];
      int seq_len_prefix = seq_lens[req_pool_id];

      int bs = req_pool_id;  // use req_pool_id to index block_table and seq_lens

      const int32_t* __restrict__ req_block_table = block_table + bs * max_block_cnt;

      TORCH_CHECK(seq_len_prefix >= 0, "prefix len < 0!");
      TORCH_CHECK(seq_len_prefix <= max_block_cnt * block_size, "seq_len out of scope!");
      TORCH_CHECK(req_pool_id < max_num_reqs, "req_pool_id out of scope!");

      // offset and size in MB
      int m = mb * BLOCK_M;
      int m_size = std::min(BLOCK_M, q_lens[bs] - m);

      if (m_size <= 0) {
        data_index_step(batch, batches, head_id, num_heads, mb, MB);
        continue;
      }

      // get query, query[cu_q_lens[bs] + m]
      const scalar_t* __restrict__ q_ptr = q + (cu_q_lens[bs] + m) * q_strideM + head_id * q_strideH;

      // init v', s' and m'
      fill_stub(v_prime, 0.f, m_size * head_size_v);
      fill_stub(s_prime, 0.f, m_size);
      fill_stub(m_prime, -std::numeric_limits<float>::infinity(), m_size);

      // compute scores with k_buffer only (no extend part)
      for (int n = 0; n < seq_len_prefix; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, seq_len_prefix - n);

        // `n_size` is K in 2nd gemm, pad to TILE_K;
        const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

        alignas(64) index_t req_to_token[BLOCK_N];
        std::string line = "*";
        for (int i = 0; i < 20; i++) {
          line += "*";
        }
        // std::cout << line << std::endl;
        // std::cout << "n_size:" << n_size << '\n';
        // std::cout << "m(q_offset): " << m << " m_size: " << m_size << '\n';
        // std::cout << "n(kv_offset): " << n << " n_size: " << n_size << '\n';

        for (int i = 0; i < n_size; i++) {
          // std::cout << "req_to_token i: " << i << " n+i: " << n + i << " block: " << (n + i) / block_size
          //           << " offset: " << (n + i) % block_size << std::endl;
          req_to_token[i] = req_block_table[(n + i) / block_size] * block_size + (n + i) % block_size;
        }

        // get key and pack
        pack_vnni<scalar_t, index_t>(
            /*    dst */ Btmp,
            /*    src */ k_buffer + head_kv_id * k_strideH,
            /*    ind */ req_to_token,
            /*      N */ n_size,
            /*      K */ head_size,
            /* ld_src */ k_strideN,
            /* ld_dst */ BLOCK_N);

        // calculate s_i <- Q @ K
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ head_size,
            /* lda   */ q_strideM,
            /* ldb   */ BLOCK_N,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ q_ptr,
            /* B     */ Btmp,
            /* C     */ s_i);

        const Vec scale_vec = Vec(scaling);
        for (int row = 0; row < m_size; ++row) {
          // s_i <- s_i * scale
          at::vec::map<float>(
              [scale_vec](Vec x) { return x * scale_vec; }, s_i + row * BLOCK_N, s_i + row * BLOCK_N, n_size);

          if (has_logit_cap) {
            at::vec::map<float>(
                [logit_cap, rlogit_cap](Vec x) { return Vec(logit_cap) * (x * Vec(rlogit_cap)).tanh(); },
                s_i + row * BLOCK_N,
                s_i + row * BLOCK_N,
                n_size);
          }

          // m_i: max value per row
          float m_i = at::vec::reduce_all<float>(
              [](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, s_i + row * BLOCK_N, n_size);
          m_i = std::max(m_i, m_prime[row]);

          // m_delta <- exp(m' - m_i)
          float m_delta = std::exp(m_prime[row] - m_i);

          // s_delta <- exp(s_i - m_i)
          at::vec::map<float>(
              [m_i](Vec x) { return (x - Vec(m_i)).exp_u20(); }, s_delta + row * BLOCK_N, s_i + row * BLOCK_N, n_size);

          // s' <- s' * m_delta + sum(s_delta)
          s_prime[row] *= m_delta;
          s_prime[row] +=
              at::vec::reduce_all<float>([](Vec& x, Vec& y) { return x + y; }, s_delta + row * BLOCK_N, n_size);

          m_prime[row] = m_i;

          // v' <- v' * m_delta
          at::vec::map<float>(
              [m_delta](Vec x) { return x * Vec(m_delta); },
              v_prime + row * head_size_v,
              v_prime + row * head_size_v,
              head_size_v);

          // pad s_delta with 0 first and then convert to scalar_t
          fill_stub(s_delta + row * BLOCK_N + n_size, 0.f, padded_n_size - n_size);
          copy_stub<scalar_t, BLOCK_N>(s_delta2 + row * BLOCK_N, s_delta + row * BLOCK_N);
        }

        // get value and pack
        pack_vnni2<scalar_t, index_t>(
            /*    dst */ Btmp,
            /*    src */ v_buffer + head_kv_id * v_strideH,
            /*    ind */ req_to_token,
            /*      K */ n_size,
            /*      N */ head_size_v,
            /* ld_src */ v_strideN,
            /* ld_dst */ head_size_v);

        // calculate V' <- s_delta @ V + V'
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ head_size_v,
            /* K     */ padded_n_size,
            /* lda   */ BLOCK_N,
            /* ldb   */ head_size_v,
            /* ldc   */ head_size_v,
            /* add_C */ true,
            /* A     */ s_delta2,
            /* B     */ Btmp,
            /* C     */ v_prime);
      }  // loop with seq_len_prefix

      // get output ptr, output[cu_q_lens[bs] + m][head_id]
      scalar_t* __restrict__ out_ptr = output + (cu_q_lens[bs] + m) * o_strideM + head_id * o_strideH;
      for (int row = 0; row < m_size; ++row) {
        float s = 1 / s_prime[row];
        copy_stub<scalar_t>(out_ptr + row * o_strideM, v_prime + row * head_size_v, s, head_size_v);

        // Store logsumexp
        int global_row = cu_q_lens[bs] + m + row;
        lse_output[global_row * num_heads + head_id] = m_prime[row] + std::log(s_prime[row]);
      }

      // move to the next index
      data_index_step(batch, batches, head_id, num_heads, mb, MB);
    }
    at::native::cpublas::brgemm_release();
  });
}
}  // namespace

// query:            [num_tokens, num_heads, head_size]
// k_buffer:         [block_cnt, block_size, num_heads, head_size]
// v_buffer:         [block_cnt, block_size, num_heads, head_size_v]
// output:           [num_tokens, num_heads, head_size]
// lse:              [num_tokens, num_heads]
// cu_q_lens:        [num_seqs] int64
// seq_lens:         [num_seqs] int64
// block_table:      [num_seqs, max_block_cnt] int64
// req_pool_indices: [num_seqs] int64
//
void decode_paged_attention_cpu(
    at::Tensor& query,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& output,
    at::Tensor& lse,
    at::Tensor& block_table,
    at::Tensor& req_pool_indices,
    at::Tensor& cu_q_lens,
    at::Tensor& seq_lens,
    double sm_scale,
    double logit_cap) {
  RECORD_FUNCTION(
      "sgl-kernel::decode_attention_cpu",
      std::vector<c10::IValue>({query, k_buffer, v_buffer, output, lse, block_table, req_pool_indices, seq_lens}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_buffer);

  CHECK_DIM(3, query);
  CHECK_DIM(4, k_buffer);
  CHECK_DIM(4, v_buffer);
  CHECK_DIM(2, lse);
  CHECK_DIM(2, block_table);
  CHECK_DIM(1, req_pool_indices);
  CHECK_DIM(1, seq_lens);

  auto head_dim = query.size(2);
  auto head_dim_v = v_buffer.size(3);
  CHECK_EQ(head_dim, k_buffer.size(3));
  CHECK_EQ(query.size(0), lse.size(0));  // num_tokens
  CHECK_EQ(query.size(1), lse.size(1));  // num_heads

  int64_t num_seqs = seq_lens.size(0);
  int64_t block_size = k_buffer.size(1);
  // number of blocks per sequence row in block_table (do NOT multiply by block_size)
  int64_t max_block_cnt = block_table.size(1);
  // total tokens available in KV buffers = block_cnt * block_size
  int64_t max_total_num_tokens = k_buffer.size(0) * k_buffer.size(1);

  int64_t num_heads = query.size(1);
  // k_buffer shape: [block_cnt, block_size, num_heads_kv, head_size]
  int64_t num_heads_kv = k_buffer.size(2);
  int64_t head_size = query.size(2);
  // v_buffer shape: [block_cnt, block_size, num_heads_kv, head_size_v]
  int64_t head_size_v = v_buffer.size(3);

  constexpr int64_t num_kv_splits = 16;

  // strides for query
  int64_t q_strideM = query.stride(0);
  int64_t q_strideH = query.stride(1);

  // strides for k_buffer and v_buffer
  int64_t k_strideN = k_buffer.stride(1);
  int64_t k_strideH = k_buffer.stride(2);

  int64_t v_strideN = v_buffer.stride(1);
  int64_t v_strideH = v_buffer.stride(2);

  // check index data types
  const auto index_dtype = block_table.scalar_type();
  TORCH_CHECK(
      block_table.scalar_type() == at::kInt,
      "prefill: expect block_table to be int32, got ",
      block_table.scalar_type());
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "prefill: expect req_lens to be int64, got ", seq_lens.scalar_type());
  TORCH_CHECK(
      req_pool_indices.scalar_type() == at::kInt,
      "prefill: expect req_pool_indices to be int32, got ",
      req_pool_indices.scalar_type());
  TORCH_CHECK(
      cu_q_lens.scalar_type() == at::kInt, "prefill: expect cu_q_lens to be int32, got ", cu_q_lens.scalar_type());

  // check if we have MLA here
  void* k_buffer_data = k_buffer.data_ptr();
  void* v_buffer_data = v_buffer.data_ptr();
  const bool is_mla = (k_buffer_data == v_buffer_data) && (num_heads_kv == 1) && (head_size == head_size_v + 64);

  // block length for k_buffer and v_buffer
  constexpr int BLOCK_N = 256;

  // buffer for packing k_cache and v_cache
  int num_threads = at::get_num_threads();
  int64_t size_per_thread = is_mla ? BLOCK_N * head_size + BLOCK_N * head_size_v : 0;
  auto buffer = at::empty({num_threads, size_per_thread}, k_buffer.options());
  auto attn_logits = at::empty({num_seqs, num_heads, num_kv_splits, head_size_v + 1}, at::kFloat);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "decode_attention_kernel", [&] {
    AT_DISPATCH_INDEX_TYPES(index_dtype, "decode_attention_indices", [&] {
      decode_paged_attention_grouped_kernel_impl<scalar_t, index_t, BLOCK_N>(
          query.data_ptr<scalar_t>(),
          (const scalar_t*)k_buffer_data,
          (const scalar_t*)v_buffer_data,
          output.data_ptr<scalar_t>(),
          attn_logits.data_ptr<float>(),
          lse.data_ptr<float>(),
          block_table.data_ptr<int32_t>(),
          req_pool_indices.data_ptr<int32_t>(),
          cu_q_lens.data_ptr<int32_t>(),
          seq_lens.data_ptr<int32_t>(),
          req_pool_indices.numel(),
          num_heads,
          num_heads_kv,
          head_size,
          head_size_v,
          num_kv_splits,
          q_strideM,
          q_strideH,
          k_strideN,
          k_strideH,
          v_strideN,
          v_strideH,
          sm_scale,
          logit_cap,
          block_size,
          max_block_cnt,
          max_total_num_tokens);
    });
  });
}

void prefill_paged_attention_cpu(
    at::Tensor& query,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& output,
    at::Tensor& lse,
    at::Tensor& block_table,
    at::Tensor& req_pool_indices,
    at::Tensor& cu_q_lens,
    at::Tensor& q_lens,
    at::Tensor& seq_lens,
    double sm_scale,
    double logit_cap) {
  RECORD_FUNCTION(
      "alayajet::prefill_attention_cpu",
      std::vector<c10::IValue>({query, k_buffer, v_buffer, output, lse, block_table, req_pool_indices, seq_lens}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_INPUT(output);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_buffer);

  CHECK_DIM(3, query);
  CHECK_DIM(3, output);
  // the query and output dim should be the same
  TORCH_CHECK(
      query.size(0) == output.size(0) && query.size(1) == output.size(1) && query.size(2) == output.size(2),
      "query and output shape mismatch: ",
      query.sizes(),
      " vs ",
      output.sizes());

  CHECK_DIM(4, k_buffer);
  CHECK_DIM(4, v_buffer);
  CHECK_DIM(2, lse);

  CHECK_DIM(2, block_table);
  CHECK_DIM(1, req_pool_indices);
  CHECK_DIM(1, seq_lens);
  CHECK_DIM(1, q_lens);
  CHECK_DIM(1, cu_q_lens);

  // check all tensor in the CPU device and last dim contiguous
  CHECK_INPUT(query);
  CHECK_INPUT(k_buffer);
  CHECK_INPUT(v_buffer);
  CHECK_INPUT(output);
  CHECK_INPUT(lse);
  CHECK_INPUT(block_table);
  CHECK_INPUT(req_pool_indices);
  CHECK_INPUT(cu_q_lens);
  CHECK_INPUT(q_lens);
  CHECK_INPUT(seq_lens);

  int num_seqs = seq_lens.size(0);
  int max_num_reqs = block_table.size(0);
  int block_size = k_buffer.size(1);                       // in tokens
  int max_context_len = block_table.size(1) * block_size;  // in tokens
  int max_total_num_tokens = k_buffer.size(0);

  int num_heads = query.size(1);
  int num_heads_kv = k_buffer.size(2);
  int head_size = query.size(2);
  int head_size_v = k_buffer.size(3);

  // strides for q_extend, k_extend and v_extend
  int q_strideM = query.stride(0);
  int q_strideH = query.stride(1);

  // strides for k_buffer and v_buffer
  int k_strideN = k_buffer.stride(1);
  int k_strideH = k_buffer.stride(2);
  int v_strideN = v_buffer.stride(1);
  int v_strideH = v_buffer.stride(2);

  // check sizes
  // MLA will skip prefix part
  const bool is_prefix_skipped = k_buffer.size(1) != num_heads_kv;

  // check index data types
  TORCH_CHECK(
      block_table.scalar_type() == at::kInt,
      "prefill: expect block_table to be int32, got ",
      block_table.scalar_type());
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "prefill: expect req_lens to be int64, got ", seq_lens.scalar_type());
  TORCH_CHECK(
      req_pool_indices.scalar_type() == at::kInt,
      "prefill: expect req_pool_indices to be int32, got ",
      req_pool_indices.scalar_type());
  TORCH_CHECK(q_lens.scalar_type() == at::kInt, "prefill: expect q_lens to be int32, got ", q_lens.scalar_type());
  TORCH_CHECK(
      cu_q_lens.scalar_type() == at::kInt, "prefill: expect cu_q_lens to be int32, got ", cu_q_lens.scalar_type());

  // block size for query seq length
  constexpr int BLOCK_M = 32;
  // block size for key/value seq length
  constexpr int BLOCK_N = 32;

  const int size_per_thread =
      /* s_i     */ BLOCK_M * BLOCK_N * sizeof(float) +
      /* v_prime */ BLOCK_M * head_size_v * sizeof(float) +
      /* s_delta */ BLOCK_M * BLOCK_N * sizeof(uint16_t) +
      /* Btmp    */ BLOCK_N * std::max(head_size, head_size_v) * sizeof(uint16_t);

  int num_threads = at::get_num_threads();
  auto buffer = at::empty({num_threads, size_per_thread}, query.options().dtype(at::kChar));

  auto index_dtype = at::kInt;
  AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "decode_attention_kernel", [&] {
    AT_DISPATCH_INDEX_TYPES(index_dtype, "decode_attention_indices", [&] {
      prefill_paged_attention_grouped_kernel_impl<scalar_t, index_t, BLOCK_M, BLOCK_N>(
          query.data_ptr<scalar_t>(),
          k_buffer.data_ptr<scalar_t>(),
          v_buffer.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          lse.data_ptr<float>(),
          block_table.data_ptr<int32_t>(),
          req_pool_indices.data_ptr<int32_t>(),
          seq_lens.data_ptr<int32_t>(),
          q_lens.data_ptr<int32_t>(),
          cu_q_lens.data_ptr<int32_t>(),
          buffer.data_ptr(),
          req_pool_indices.numel(),
          num_heads,
          num_heads_kv,
          head_size,
          head_size_v,
          q_strideM,
          q_strideH,
          k_strideN,
          k_strideH,
          v_strideN,
          v_strideH,
          sm_scale,
          logit_cap,
          max_num_reqs,
          block_table.size(1),
          q_lens.max().item<int>(),
          block_size,
          size_per_thread);
    });
  });
}

void flash_paged_attention_cpu(
    at::Tensor& query,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& output,
    at::Tensor& lse,
    at::Tensor& block_table,
    // cumulative q lengths per sequence (start offsets)
    at::Tensor& cu_q_lens,
    at::Tensor& seq_lens,
    at::Tensor& q_lens,
    double sm_scale,
    double logit_cap) {
  RECORD_FUNCTION(
      "alaya-jet::flash_attention_cpu",
      std::vector<c10::IValue>({query, k_buffer, v_buffer, output, lse, block_table, cu_q_lens, seq_lens}));
  std::vector<int32_t> prefill_indice;
  std::vector<int32_t> decode_indice;
  prefill_indice.reserve(cu_q_lens.size(0));
  decode_indice.reserve(cu_q_lens.size(0));

  for (int i = 0; i < q_lens.size(0); i++) {
    if (q_lens[i].item<int32_t>() > 1) {
      prefill_indice.push_back(i);
    } else {
      decode_indice.push_back(i);
    }
  }
  auto req_pool_indices_prefill = at::tensor(prefill_indice, at::kInt);
  auto req_pool_indices_decode = at::tensor(decode_indice, at::kInt);

  if (req_pool_indices_prefill.numel() > 0) {
    prefill_paged_attention_cpu(
        query,
        k_buffer,
        v_buffer,
        output,
        lse,
        block_table,
        req_pool_indices_prefill,
        cu_q_lens,
        q_lens,
        seq_lens,
        sm_scale,
        logit_cap);
  }
  if (req_pool_indices_decode.numel() > 0)
    decode_paged_attention_cpu(
        query,
        k_buffer,
        v_buffer,
        output,
        lse,
        block_table,
        req_pool_indices_decode,
        cu_q_lens,
        seq_lens,
        sm_scale,
        logit_cap);
}
