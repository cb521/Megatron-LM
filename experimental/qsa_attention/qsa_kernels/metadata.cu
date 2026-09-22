// Exact per-query token membership with bounded block compaction.
// Repeated IDs preserve their original multiplicity through the occurrence-preserving chunk path.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>


template<int CAP>
struct ChunkStorage {
  using Sort = cub::BlockRadixSort<uint64_t, 128, CAP / 128, int>;
  using Scan = cub::BlockScan<int, 128>;
  union { typename Sort::TempStorage sort; typename Scan::TempStorage scan; } temp;
  int base[CAP];
  unsigned bits[CAP];
  uint64_t ordered[CAP];
};

template<int G, int CAP>
struct UnionStorage {
  using Sort = cub::BlockRadixSort<uint32_t, 128, CAP / 128, int>;
  using Scan = cub::BlockScan<int, 128>;
  union { typename Sort::TempStorage sort; typename Scan::TempStorage scan; } temp;
  // Zero is empty. A live key encodes ((physical_block + 1) << 1),
  // with bit zero recording whether the old path had a canonical chunk.
  uint32_t compact_keys[CAP];
  int compact_slots[CAP];
  int failed, count, canonical_count, opaque_count;
};

template<int G, int CAP>
union MetadataStorage {
  ChunkStorage<CAP> legacy;
  UnionStorage<G, CAP> exact;
};
template<int G, int CAP, bool CANONICAL_ONLY = false>
__device__ __forceinline__ bool pack_chunks(const int* ids, const int* valid, int* out_base,
                           int* out_bits, int* counts, int sq, int slots, int skv, ChunkStorage<CAP>& storage) {
  constexpr int NT = 128;
  constexpr int ITEMS = CAP / NT;
  using Sort = cub::BlockRadixSort<uint64_t, NT, ITEMS, int>;
  using Scan = cub::BlockScan<int, NT>;
  auto& tmp = storage.temp;
  auto* base = storage.base;
  auto* bits = storage.bits;
  auto* ordered = storage.ordered;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int chunks = (slots + 31) / 32;
  const int candidates = G * chunks;
  const int ng = (sq + G - 1) / G;
  const int batch = blockIdx.y, group = blockIdx.x;
  const int out_row = batch * ng + group;
  for (int i = tid; i < CAP; i += NT) {
    base[i] = -1;
    bits[i] = 0;
    out_base[out_row * CAP + i] = 0;
    #pragma unroll
    for (int g = 0; g < G; ++g) out_bits[(out_row * CAP + i) * G + g] = 0;
  }
  __syncthreads();
  for (int ci = warp; ci < candidates; ci += NT / 32) {
    const int q = group * G + ci / chunks;
    const int slot = (ci % chunks) * 32 + lane;
    int token = 0;
    bool live = false;
    if (q < sq && slot < slots) {
      const int address = (batch * sq + q) * slots + slot;
      live = valid[address] != 0;
      token = ids[address];
    }
    const unsigned mask = __ballot_sync(0xffffffff, live);
    const int first = mask ? __ffs(mask) - 1 : 0;
    const int first_token = __shfl_sync(0xffffffff, token, first);
    const int start = first_token - first;
    const bool canonical = __all_sync(0xffffffff, !live || token - lane == start)
                         && start >= 0 && start % 32 == 0;
    if (lane == 0) {
      bits[ci] = mask;
      base[ci] = canonical ? start : -ci - 1;
    }
  }
  __syncthreads();
  if constexpr (CANONICAL_ONLY) {
    bool opaque = false;
    for (int ci = tid; ci < candidates; ci += NT)
      opaque |= bits[ci] != 0 && base[ci] < 0;
    // Aligned chunks with no repeated block per query already encode the
    // physical-block union; the second check below verifies that condition.
    if (__syncthreads_or(opaque)) return false;
  }
  uint64_t keys[ITEMS];
  int values[ITEMS];
  bool repeated_block = false;
  const uint64_t opaque_begin = ((uint64_t(skv) + 31) / 32 + 1) * (chunks + 1);
  #pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int ci = tid * ITEMS + i;
    keys[i] = UINT64_MAX;
    values[i] = ci;
    if (ci < candidates && bits[ci]) {
      if (base[ci] >= 0) {
        int occurrence = 0;
        for (int prev = (ci / chunks) * chunks; prev < ci; ++prev)
          occurrence += bits[prev] && base[prev] == base[ci];
        if constexpr (CANONICAL_ONLY) repeated_block |= occurrence != 0;
        keys[i] = uint64_t(base[ci] / 32) * (chunks + 1) + occurrence;
      } else {
        keys[i] = opaque_begin + ci;
      }
    }
  }
  if constexpr (CANONICAL_ONLY) {
    // Separate chunks of one query may select disjoint parts of the same
    // physical block. The membership union must combine those as well.
    if (__syncthreads_or(repeated_block)) return false;
  }
  // Only meaningful radix bits participate. The all-ones sentinel stays last.
  const int radix_end = 64 - __clzll(opaque_begin + candidates);
  Sort(tmp.sort).Sort(keys, values, 0, radix_end);
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < ITEMS; ++i) ordered[tid * ITEMS + i] = keys[i];
  __syncthreads();
  int heads[ITEMS], ranks[ITEMS];
  #pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int index = tid * ITEMS + i;
    heads[i] = keys[i] != UINT64_MAX && (index == 0 || ordered[index - 1] != keys[i]);
  }
  int total;
  Scan(tmp.scan).InclusiveSum(heads, ranks, total);
  #pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    if (keys[i] != UINT64_MAX) {
      const int ci = values[i], dest = out_row * CAP + ranks[i] - 1;
      if (heads[i]) out_base[dest] = base[ci];
      // Equal keys contain at most one chunk from each query.
      out_bits[dest * G + ci / chunks] = static_cast<int>(bits[ci]);
    }
  }
  if (tid == 0) counts[out_row] = total;
  return true;
}


template<int G, int CAP>
__global__ void pack_routes_union(const int* ids, const int* valid, int* out_base,
                                 int* out_bits, int* counts,
                                 int sq, int slots, int skv, int hash_size) {
  constexpr int NT = 128;
  const int HASH = hash_size;
  constexpr int ITEMS = CAP / NT;
  using Sort = typename UnionStorage<G, CAP>::Sort;
  using Scan = typename UnionStorage<G, CAP>::Scan;
  extern __shared__ __align__(16) unsigned char scratch[];
  auto& storage = *reinterpret_cast<MetadataStorage<G, CAP>*>(scratch);
  if (pack_chunks<G, CAP, true>(ids, valid, out_base, out_bits, counts,
                                  sq, slots, skv, storage.legacy)) return;
  __syncthreads();
  auto& shared = storage.exact;
  auto* hash_keys = reinterpret_cast<int*>(scratch + sizeof(UnionStorage<G, CAP>));
  auto* hash_members = reinterpret_cast<unsigned (*)[G]>(hash_keys + HASH);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int chunks = (slots + 31) / 32;
  const int candidates = G * chunks;
  const int ng = (sq + G - 1) / G;
  const int batch = blockIdx.y, group = blockIdx.x;
  const int out_row = batch * ng + group;
  for (int i = tid; i < HASH; i += NT) {
    hash_keys[i] = 0;
    #pragma unroll
    for (int g = 0; g < G; ++g) hash_members[i][g] = 0;
  }
  if (tid == 0) {
    shared.failed = 0;
    shared.count = 0;
    shared.canonical_count = 0;
    shared.opaque_count = 0;
  }
  __syncthreads();

  for (int ci = warp; ci < candidates; ci += NT / 32) {
    const int local_q = ci / chunks;
    const int q = group * G + local_q;
    const int slot = (ci % chunks) * 32 + lane;
    int token = 0;
    bool live = false;
    if (q < sq && slot < slots) {
      const int address = (batch * sq + q) * slots + slot;
      live = valid[address] != 0;
      if (live) token = ids[address];
    }
    const unsigned mask = __ballot_sync(0xffffffffu, live);
    const int first = mask ? __ffs(mask) - 1 : 0;
    const int start = __shfl_sync(0xffffffffu, token, first) - first;
    const bool canonical = __all_sync(0xffffffffu, !live || token - lane == start)
                           && start >= 0 && start % 32 == 0;
    if (lane == 0 && mask && !canonical) atomicAdd(&shared.opaque_count, 1);

    // A warp owns one query. Combine its selected token bits by physical
    // block, independently of their positions in the supplied route list.
    const int block = live ? token / 32 : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, block);
    const unsigned member_bits = __reduce_or_sync(peers, live ? 1u << (token & 31) : 0u);
    if (live && lane == __ffs(peers) - 1) {
      if (__popc(member_bits) != __popc(peers)) atomicExch(&shared.failed, 1);
      const int encoded = (block + 1) << 1;
      int location = (static_cast<unsigned>(block) * 2654435761u) & (HASH - 1);
      bool inserted = false;
      // Bounded probing prevents malformed/high-fragmentation inputs from
      // creating an unbounded metadata loop. The original route is retained.
      #pragma unroll 1
      for (int probe = 0; probe < 32; ++probe) {
        const int old = atomicCAS(&hash_keys[location], 0, encoded | int(canonical));
        if (old == 0 || (old & ~1) == encoded) {
          if (old == 0) {
            atomicAdd(&shared.count, 1);
            if (canonical) atomicAdd(&shared.canonical_count, 1);
          } else if (canonical) {
            const int marked = atomicOr(&hash_keys[location], 1);
            if (!(marked & 1)) atomicAdd(&shared.canonical_count, 1);
          }
          const unsigned previous = atomicOr(&hash_members[location][local_q], member_bits);
          if (previous & member_bits) atomicExch(&shared.failed, 1);
          inserted = true;
          break;
        }
        location = (location + 1) & (HASH - 1);
      }
      if (!inserted) atomicExch(&shared.failed, 1);
    }
  }
  __syncthreads();

  // A bool membership word cannot encode multiple occurrences in one query.
  // Also avoid expanding fragmented token lists into more physical tiles.
  const bool fallback = shared.failed || shared.count > CAP ||
                        shared.count > shared.canonical_count + shared.opaque_count;
  if (fallback) {
    __syncthreads();
    pack_chunks<G, CAP>(ids, valid, out_base, out_bits, counts,
                         sq, slots, skv, storage.legacy);
    return;
  }
  const int total = shared.count;
  int local_count = 0;
  for (int slot = tid; slot < HASH; slot += NT) local_count += hash_keys[slot] != 0;
  int output_begin = 0;
  Scan(shared.temp.scan).ExclusiveSum(local_count, output_begin);
  __syncthreads();
  int local_offset = 0;
  for (int slot = tid; slot < HASH; slot += NT) {
    const int encoded = hash_keys[slot];
    if (encoded) {
      shared.compact_keys[output_begin + local_offset] = (encoded >> 1) - 1;
      shared.compact_slots[output_begin + local_offset] = slot;
      ++local_offset;
    }
  }
  __syncthreads();
  uint32_t keys[ITEMS];
  int hash_slots[ITEMS];
  #pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int row = tid * ITEMS + i;
    keys[i] = row < total ? shared.compact_keys[row] : UINT32_MAX;
    hash_slots[i] = row < total ? shared.compact_slots[row] : 0;
  }
  const unsigned physical_blocks = (static_cast<unsigned>(skv) + 31u) / 32u;
  const int radix_end = 32 - __clz(physical_blocks);
  Sort(shared.temp.sort).Sort(keys, hash_slots, 0, radix_end);
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int row = tid * ITEMS + i;
    const int dest = out_row * CAP + row;
    out_base[dest] = row < total ? static_cast<int>(keys[i] * 32u) : 0;
    #pragma unroll
    for (int g = 0; g < G; ++g)
      out_bits[dest * G + g] = row < total ? static_cast<int>(hash_members[hash_slots[i]][g]) : 0;
  }
  if (tid == 0) counts[out_row] = total;
}

template<int G, int CAP>
void launch_routes(const dim3& grid, cudaStream_t stream,
                   const int* ids, const int* valid, int* bases, int* bits, int* counts,
                   int sq, int slots, int skv) {
  const unsigned physical_blocks = (static_cast<unsigned>(skv) + 31u) / 32u;
  int hash_size = 1;
  while (hash_size < 2 * CAP && hash_size < physical_blocks) hash_size <<= 1;
  const int union_bytes = sizeof(UnionStorage<G, CAP>) + hash_size * (G + 1) * sizeof(int);
  const int bytes = union_bytes > sizeof(ChunkStorage<CAP>) ? union_bytes : sizeof(ChunkStorage<CAP>);
  if (bytes > 48 * 1024) {
    // Every host thread sets the same ceiling for this specialization, so
    // plans with different KV lengths cannot lower one another's limit.
    constexpr int max_union_bytes = sizeof(UnionStorage<G, CAP>) + 2 * CAP * (G + 1) * sizeof(int);
    constexpr int max_bytes = max_union_bytes > sizeof(ChunkStorage<CAP>)
                                  ? max_union_bytes : sizeof(ChunkStorage<CAP>);
    static thread_local int configured_device = -1;
    int device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    if (device != configured_device) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(pack_routes_union<G, CAP>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, max_bytes));
      configured_device = device;
    }
  }
  pack_routes_union<G, CAP><<<grid, 128, bytes, stream>>>(
      ids, valid, bases, bits, counts, sq, slots, skv, hash_size);
}
template<int G>
void dispatch_capacity(int cap, const at::Tensor& ids, const at::Tensor& valid,
                      at::Tensor& bases, at::Tensor& bits, at::Tensor& counts, int skv) {
  const dim3 grid((ids.size(1) + G - 1) / G, ids.size(0));
  auto stream = c10::cuda::getCurrentCUDAStream(ids.get_device()).stream();
  #define LAUNCH(C) launch_routes<G, C>(grid, stream, ids.data_ptr<int>(), valid.data_ptr<int>(), bases.data_ptr<int>(), bits.data_ptr<int>(), counts.data_ptr<int>(), ids.size(1), ids.size(2), skv)
  switch (cap) {
    case 128: LAUNCH(128); break;
    case 256: LAUNCH(256); break;
    case 512: LAUNCH(512); break;
    case 1024: LAUNCH(1024); break;
    default: TORCH_CHECK(false, "unsupported metadata capacity");
  }
  #undef LAUNCH
}

void build_metadata(const at::Tensor& ids, const at::Tensor& valid,
                    at::Tensor bases, at::Tensor bits, at::Tensor counts, int64_t group, int64_t skv) {
  TORCH_CHECK(ids.is_cuda() && valid.is_cuda() && ids.scalar_type() == at::kInt && valid.scalar_type() == at::kInt);
  TORCH_CHECK(ids.is_contiguous() && valid.is_contiguous() && ids.sizes() == valid.sizes());
  TORCH_CHECK(ids.dim() == 3 && ids.size(0) > 0 && ids.size(1) > 0 && ids.size(2) > 0,
              "ids/valid must have nonempty shape [B,Q,S]");
  TORCH_CHECK(valid.device() == ids.device() && skv > 0 && skv <= INT_MAX,
              "invalid route device or KV length");
  TORCH_CHECK(group == 1 || group == 2 || group == 4 || group == 5,
              "unsupported query group");
  const int64_t groups = (ids.size(1) + group - 1) / group;
  TORCH_CHECK(bases.dim() == 3 && bits.dim() == 4 && counts.dim() == 2,
              "invalid metadata output ranks");
  TORCH_CHECK(bases.size(0) == ids.size(0) && bases.size(1) == groups &&
              bits.size(0) == ids.size(0) && bits.size(1) == groups &&
              bits.size(2) == bases.size(2) && bits.size(3) == group &&
              counts.size(0) == ids.size(0) && counts.size(1) == groups,
              "invalid metadata output shapes");
  for (const auto& tensor : {bases, bits, counts}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == ids.device() &&
                tensor.scalar_type() == at::kInt && tensor.is_contiguous(),
                "metadata outputs must be contiguous int32 on the route device");
    TORCH_CHECK(tensor.numel() <= INT_MAX, "metadata exceeds int32 indexing range");
  }
  TORCH_CHECK(ids.numel() <= INT_MAX && ids.size(0) <= 65535,
              "route tensor exceeds supported grid/indexing range");
  const c10::cuda::CUDAGuard guard(ids.device());
  const int cap = bases.size(2);
  TORCH_CHECK(cap >= group * ((ids.size(2) + 31) / 32));
  switch (group) {
    case 1: dispatch_capacity<1>(cap, ids, valid, bases, bits, counts, skv); break;
    case 2: dispatch_capacity<2>(cap, ids, valid, bases, bits, counts, skv); break;
    case 4: dispatch_capacity<4>(cap, ids, valid, bases, bits, counts, skv); break;
    case 5: dispatch_capacity<5>(cap, ids, valid, bases, bits, counts, skv); break;
    default: TORCH_CHECK(false, "unsupported query group");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY(qsa_routes_clean_v3, m) {
  m.def("build(Tensor ids, Tensor valid, Tensor(a!) bases, Tensor(b!) bits, Tensor(c!) counts, int group, int skv) -> ()");
}
TORCH_LIBRARY_IMPL(qsa_routes_clean_v3, CUDA, m) { m.impl("build", &build_metadata); }
