/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// PCIe Ordering Stress Tests for NIC HW Flush Validation
//
// RCCL uses RDMA WRITE_IMM to deliver data into GPU memory via GDR. After the
// HCA signals CQE completion, the data may not yet be globally visible in GPU
// memory due to PCIe posted-write ordering. Three flush modes exist:
//
//   1. NIC HW flush only  (NCCL_GDR_FLUSH_DISABLE=1)
//      No explicit SW flush from CPU — relies entirely on the NIC hardware to
//      ensure PCIe write ordering before CQE. This is the primary mode under
//      test: it validates that the NIC's HW flush is implemented correctly.
//   2. SW flush: single RDMA READ from data buffer
//      (NCCL_GDR_FLUSH_DISABLE=0, RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0)
//      CPU-side explicit flush fallback — issues an RDMA READ that forces the
//      PCIe memory controller to drain prior posted writes.
//   3. SW flush: WRITE+READ to dedicated GPU dummy buffer
//      (NCCL_GDR_FLUSH_DISABLE=0, RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=1)
//      Another CPU-side explicit flush variant.
//
// These tests blast large RDMA WRITE_IMM messages into GPU buffers across
// multiple concurrent connections, then verify every byte using GPU CU loads
// (not DMA), to validate that the NIC's HW flush correctly ensures PCIe write
// ordering before CQE delivery. Failures in mode 1 indicate a NIC HW flush bug.
//
// Run examples:
//   NCCL_GDR_FLUSH_DISABLE=1 mpirun -np 4 ./rccl-UnitTestsMPI --gtest_filter='*PCIeOrdering*'
//   NCCL_GDR_FLUSH_DISABLE=0 RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 mpirun -np 4 ...
//   NCCL_GDR_FLUSH_DISABLE=0 RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=1 mpirun -np 4 ...

#include "NetIbMPITestBase.hpp"
#include <unistd.h>  // usleep — for coherency diagnostic delayed re-reads

#ifdef MPI_TESTS_ENABLED

// Force the ROCm IB plugin (bypasses NCCL_NET env var selection).
extern ncclNet_t rocmNetIb;

// Minimum 2 nodes required — rocmNetIb on AINIC does not support single-node loopback.
static constexpr int kMinTwoNodes = 2;

// Timeout for large PCIe-ordered transfers (16 MB over RDMA).
static constexpr int kPCIeTimeoutMs = 60000;

// Default iteration count; overridable via PCIE_ORDER_ITERS env var.
static int getPCIeIters()
{
    const char* env = getenv("PCIE_ORDER_ITERS");
    return (env && env[0]) ? std::max(1, atoi(env)) : 200;
}

// Print verification mode once (rank 0 only).
static void printVerifyModeOnce()
{
    static bool printed = false;
    if (!printed)
    {
        printed = true;
        const char* env = getenv("PCIE_ORDER_GPU_VERIFY");
        bool gpu = (!env || env[0] == '\0' || atoi(env) != 0);
        std::cout << "[PCIeOrdering] Verification mode: "
                  << (gpu ? "GPU CU kernel (CU load path)"
                          : "DMA (hipMemcpy to host)")
                  << std::endl;
    }
}

// All available message sizes: 64KB, 256KB, 1MB, 4MB, 16MB.
static const std::vector<size_t> kAllMessageSizes = {
    64   * 1024,
    256  * 1024,
    1    * 1024 * 1024,
    4    * 1024 * 1024,
    16   * 1024 * 1024,
};

// Parse size string with optional KB/MB suffix (e.g., "256KB", "4MB", "65536").
// Returns size in bytes.
static size_t parseSizeString(const char* str)
{
    char* end = nullptr;
    size_t val = strtoull(str, &end, 10);
    if (end && (*end == 'k' || *end == 'K')) val *= 1024;
    else if (end && (*end == 'm' || *end == 'M')) val *= 1024 * 1024;
    return val;
}

// Get filtered message sizes based on PCIE_ORDER_MSG_MIN / PCIE_ORDER_MSG_MAX.
// Defaults: min=64KB, max=16MB (i.e., all sizes).
static std::vector<size_t> getPCIeMessageSizes()
{
    size_t minSize = 64 * 1024;
    size_t maxSize = 16 * 1024 * 1024;

    const char* envMin = getenv("PCIE_ORDER_MSG_MIN");
    const char* envMax = getenv("PCIE_ORDER_MSG_MAX");
    if (envMin && envMin[0]) minSize = parseSizeString(envMin);
    if (envMax && envMax[0]) maxSize = parseSizeString(envMax);

    std::vector<size_t> sizes;
    for (size_t s : kAllMessageSizes) {
        if (s >= minSize && s <= maxSize)
            sizes.push_back(s);
    }
    return sizes;
}

// =====================================================================
//  GPU-side buffer verification
//
//  A HIP kernel reads VRAM via the CU load path (L2 → VRAM), NOT the
//  DMA engine used by hipMemcpy.  This exercises the exact read path
//  that a real GPU kernel would use after NIC RDMA writes land in VRAM,
//  catching PCIe coherency issues that a DMA-based check would miss.
// =====================================================================

struct GpuVerifyResult
{
    int      mismatchFound;   // 0 = ok, 1 = at least one error
    size_t   firstErrIdx;     // byte offset of first mismatch
    uint8_t  expected;        // expected value at firstErrIdx
    uint8_t  actual;          // actual value at firstErrIdx
    uint64_t totalErrors;     // total mismatching bytes
};

__global__ void gpuVerifyBytePatternKernel(const uint8_t* __restrict__ data,
                                           size_t         numBytes,
                                           int            seed,
                                           GpuVerifyResult* result)
{
    size_t idx    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t i = idx; i < numBytes; i += stride)
    {
        uint8_t exp = static_cast<uint8_t>((seed + i) % 256);
        uint8_t act = data[i];
        if (act != exp)
        {
            atomicAdd(&result->totalErrors, 1ULL);
            // Record the first mismatch (exactly once across all threads)
            if (atomicCAS(&result->mismatchFound, 0, 1) == 0)
            {
                result->firstErrIdx = i;
                result->expected    = exp;
                result->actual      = act;
            }
        }
    }
}

/**
 * @brief Verify a device buffer using GPU CUs (not DMA).
 *
 * Launches a HIP kernel that reads every byte of @p device_buffer via
 * the normal CU load path and compares against the pattern
 * (seed + i) % 256.
 *
 * @return true if all bytes match, false on mismatch or HIP error.
 */
static bool verifyBufferOnGPU(const void* device_buffer,
                              size_t      numBytes,
                              int         seed,
                              size_t*     errIdx    = nullptr,
                              uint8_t*    errExp    = nullptr,
                              uint8_t*    errGot    = nullptr,
                              uint64_t*   totalErrs = nullptr)
{
    if (!device_buffer || numBytes == 0)
        return false;

    GpuVerifyResult* d_result = nullptr;
    hipError_t err = hipMalloc(&d_result, sizeof(GpuVerifyResult));
    if (err != hipSuccess) return false;

    err = hipMemset(d_result, 0, sizeof(GpuVerifyResult));
    if (err != hipSuccess) { hipFree(d_result); return false; }

    const int threadsPerBlock = 256;
    const int maxBlocks       = 1024;
    int blocks = std::min(maxBlocks,
                          static_cast<int>((numBytes + threadsPerBlock - 1) / threadsPerBlock));

    gpuVerifyBytePatternKernel<<<blocks, threadsPerBlock>>>(
        static_cast<const uint8_t*>(device_buffer), numBytes, seed, d_result);

    err = hipDeviceSynchronize();
    if (err != hipSuccess) { hipFree(d_result); return false; }

    GpuVerifyResult h_result = {};
    err = hipMemcpy(&h_result, d_result, sizeof(GpuVerifyResult), hipMemcpyDeviceToHost);
    hipFree(d_result);
    if (err != hipSuccess) return false;

    if (h_result.mismatchFound)
    {
        if (errIdx)    *errIdx    = h_result.firstErrIdx;
        if (errExp)    *errExp    = h_result.expected;
        if (errGot)    *errGot    = h_result.actual;
        if (totalErrs) *totalErrs = h_result.totalErrors;
        return false;
    }
    return true;
}

/**
 * @brief Verify a device buffer using DMA (hipMemcpy to host + CPU compare).
 *
 * Falls back to the DMA engine read path.  This will NOT catch PCIe
 * coherency issues between NIC writes and GPU CU reads, but is useful
 * as a baseline to confirm data actually arrived in VRAM.
 */
static bool verifyBufferDMA(const void* device_buffer,
                            size_t      numBytes,
                            int         seed,
                            size_t*     errIdx    = nullptr,
                            uint8_t*    errExp    = nullptr,
                            uint8_t*    errGot    = nullptr,
                            uint64_t*   totalErrs = nullptr)
{
    if (!device_buffer || numBytes == 0)
        return false;

    std::vector<uint8_t> host_data(numBytes);
    hipError_t err = hipMemcpy(host_data.data(), device_buffer,
                               numBytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) return false;

    uint64_t errors = 0;
    bool firstRecorded = false;
    for (size_t i = 0; i < numBytes; i++)
    {
        uint8_t exp = static_cast<uint8_t>((seed + i) % 256);
        if (host_data[i] != exp)
        {
            errors++;
            if (!firstRecorded)
            {
                firstRecorded = true;
                if (errIdx) *errIdx = i;
                if (errExp) *errExp = exp;
                if (errGot) *errGot = host_data[i];
            }
        }
    }
    if (totalErrs) *totalErrs = errors;
    return (errors == 0);
}

/**
 * @brief Check PCIE_ORDER_GPU_VERIFY env var (default: 1 = GPU CU path).
 */
static bool useGpuVerify()
{
    static int cached = -1;
    if (cached < 0)
    {
        const char* env = getenv("PCIE_ORDER_GPU_VERIFY");
        cached = (!env || env[0] == '\0' || atoi(env) != 0) ? 1 : 0;
    }
    return cached != 0;
}

/**
 * @brief Delayed re-read diagnostic to confirm PCIe coherency issue.
 *
 * Called only when the initial verification fails.  Three-phase check:
 *
 *  Phase 1: Delayed GPU CU re-reads at 10ms, 100ms, 1000ms.
 *    - If a re-read passes → PCIe coherency/ordering issue confirmed
 *      (NIC wrote data but it wasn't globally visible at CQE time).
 *
 *  Phase 2: If all GPU CU re-reads fail, do a DMA cross-check (hipMemcpy
 *    to host). This uses the DMA engine read path instead of CU loads.
 *    - DMA passes but CU fails → true CU vs DMA coherency split.
 *    - DMA also fails → data never arrived in VRAM at all (transfer issue,
 *      not a coherency issue).
 */
static void coherencyDiagnostic(const void* device_buffer,
                                size_t      numBytes,
                                int         seed,
                                uint64_t    origTotalErrs)
{
    // Phase 1: Delayed GPU CU re-reads
    static const int delays_ms[] = {10, 100, 1000};

    for (int delay : delays_ms)
    {
        usleep(delay * 1000);

        size_t   retryIdx   = 0;
        uint8_t  retryExp   = 0, retryGot = 0;
        uint64_t retryErrs  = 0;

        bool retryOk = verifyBufferOnGPU(device_buffer, numBytes, seed,
                                          &retryIdx, &retryExp, &retryGot, &retryErrs);
        if (retryOk)
        {
            std::cout << "  ** COHERENCY CONFIRMED: GPU CU re-read after "
                      << delay << "ms PASSED — data is now visible in VRAM.\n"
                      << "     Original mismatch: " << origTotalErrs
                      << " bytes wrong. This is a PCIe ordering issue." << std::endl;
            return;
        }
        else
        {
            std::cout << "  ** GPU CU re-read after " << delay << "ms: still "
                      << retryErrs << "/" << numBytes << " bytes wrong"
                      << " (original: " << origTotalErrs << ")" << std::endl;
        }
    }

    // Phase 2: DMA cross-check — read via hipMemcpy (DMA engine) instead of CU loads
    {
        size_t   dmaIdx  = 0;
        uint8_t  dmaExp  = 0, dmaGot = 0;
        uint64_t dmaErrs = 0;

        bool dmaOk = verifyBufferDMA(device_buffer, numBytes, seed,
                                      &dmaIdx, &dmaExp, &dmaGot, &dmaErrs);
        if (dmaOk)
        {
            std::cout << "  ** DMA CROSS-CHECK: hipMemcpy read PASSED — data IS in VRAM "
                      << "but GPU CU cannot see it.\n"
                      << "     This is a CU vs DMA read-path coherency issue."
                      << std::endl;
        }
        else
        {
            std::cout << "  ** DMA CROSS-CHECK: hipMemcpy read also FAILED — "
                      << dmaErrs << "/" << numBytes << " bytes wrong"
                      << " (firstErr @" << dmaIdx
                      << " expected=0x" << std::hex << (int)dmaExp
                      << " actual=0x" << (int)dmaGot << std::dec << ").\n"
                      << "     Data never fully arrived in VRAM — this is a "
                      << "transfer/completion issue, not a coherency issue."
                      << std::endl;
        }
    }
}

/**
 * @brief Unified verification dispatcher.
 *
 * Calls GPU CU kernel or DMA-based verification based on
 * PCIE_ORDER_GPU_VERIFY env var (1 = GPU CU [default], 0 = DMA).
 *
 * On failure, runs coherency diagnostic: delayed GPU CU re-reads to
 * determine whether the mismatch is a PCIe ordering issue (data eventually
 * becomes visible) or something else.
 */
static bool verifyBuffer(const void* device_buffer,
                         size_t      numBytes,
                         int         seed,
                         size_t*     errIdx    = nullptr,
                         uint8_t*    errExp    = nullptr,
                         uint8_t*    errGot    = nullptr,
                         uint64_t*   totalErrs = nullptr)
{
    bool ok;
    if (useGpuVerify())
        ok = verifyBufferOnGPU(device_buffer, numBytes, seed,
                               errIdx, errExp, errGot, totalErrs);
    else
        ok = verifyBufferDMA(device_buffer, numBytes, seed,
                             errIdx, errExp, errGot, totalErrs);

    if (!ok)
    {
        uint64_t errs = totalErrs ? *totalErrs : 0;
        coherencyDiagnostic(device_buffer, numBytes, seed, errs);
    }

    return ok;
}

// =====================================================================
//  Connection warm-up constants
//
//  QPs may require a small initial transfer to fully prime the RDMA path
//  (especially with GDR).  Without warm-up, the first real transfer on a
//  freshly established QP can report CQE completion before all data has
//  actually landed in GPU VRAM — a "first-use" issue observed on AINIC.
//  A single small send/recv per connection resolves this.
// =====================================================================

static constexpr size_t  kWarmupSize = 4096;
static constexpr int     kWarmupTag  = 9999;

// =====================================================================
//  Test 1: PCIeOrderingAllGatherFanIn
//
//  All N-1 sender ranks blast WRITE_IMM into rank 0's GPU buffer
//  concurrently, each writing to its own segment (like allgather).
//  Maximum PCIe pressure on one GPU's memory controller.
// =====================================================================
TEST_F(NetIbMPITest, PCIeOrderingAllGatherFanIn)
{
    const int nranks = MPIEnvironment::world_size;
    ASSERT_TRUE(validateTestPrerequisites(
        /*min_processes=*/kMinFourProcesses,
        /*max_processes=*/0,
        /*require_power_of_two=*/false,
        /*min_nodes=*/kMinTwoNodes,
        /*max_nodes=*/kNoNodeLimit))
        << "Test requires at least " << kMinFourProcesses
        << " MPI processes across at least " << kMinTwoNodes << " nodes";

    const int rank = MPIEnvironment::world_rank;

    // Force ROCm IB plugin
    net_ = &rocmNetIb;
    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    // Check GDR support
    ncclNetProperties_t props;
    ASSERT_EQ(GetDeviceProperties(0, &props), ncclSuccess);
    if (!(props.ptrSupport & NCCL_PTR_CUDA)) {
        GTEST_SKIP() << "No GPU Direct RDMA support — skipping PCIe ordering test";
    }

    // Build sender list: all ranks except rank 0
    std::vector<int> senderRanks;
    for (int r = 1; r < nranks; r++)
        senderRanks.push_back(r);

    // Setup fan-in: N-1 senders → rank 0
    std::vector<DirectedConnection> conns;
    SetupFanIn(/*dev=*/0, /*receiverRank=*/0, senderRanks, conns);

    const int numSenders = static_cast<int>(senderRanks.size());
    const size_t maxMsgSize = kAllMessageSizes.back();
    const size_t maxTotalRecvSize = maxMsgSize * numSenders;

    // ── GPU buffer allocation ──────────────────────────────────────
    // Rank 0: one big receive buffer (numSenders segments)
    // Senders: one send buffer sized to max message
    void* gpuBuf = nullptr;
    size_t gpuBufSize = (rank == 0) ? maxTotalRecvSize : maxMsgSize;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&gpuBuf, gpuBufSize));
    auto gpuGuard = makeDeviceBufferAutoGuard(gpuBuf);

    // ── Register MR once at max size for each connection ───────────
    // Each sender registers its send buffer against its sendComm.
    // Rank 0 registers the full recv buffer against each recvComm.
    std::vector<void*> mhandles(conns.size(), nullptr);
    for (size_t i = 0; i < conns.size(); i++) {
        void* comm = nullptr;
        if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
        if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
        if (comm) {
            ASSERT_EQ(RegisterMemory(comm, gpuBuf, gpuBufSize, NCCL_PTR_CUDA, &mhandles[i]),
                      ncclSuccess);
        }
    }
    auto mrCleanup = makeScopeGuard([&]() {
        for (size_t i = 0; i < conns.size(); i++) {
            if (!mhandles[i]) continue;
            void* comm = nullptr;
            if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
            if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
            if (comm) DeregisterMemory(comm, mhandles[i]);
        }
    });

    // ── Warm up: prime all fan-in connections ────────────────────
    {
        MPI_Barrier(MPI_COMM_WORLD);
        std::vector<void*> wRecvReqs(numSenders, nullptr);
        if (rank == 0) {
            for (int s = 0; s < numSenders; s++) {
                char* segPtr = static_cast<char*>(gpuBuf) + s * kWarmupSize;
                void*  bufs[1]    = {segPtr};
                size_t sizes[1]   = {kWarmupSize};
                int    tags[1]    = {kWarmupTag};
                void*  handles[1] = {mhandles[s]};
                PostRecv(conns[s].recvComm, 1, bufs, sizes, tags, handles,
                         &wRecvReqs[s]);
            }
        }
        void* wSendReq = nullptr;
        if (rank != 0) {
            int myConnIdx = rank - 1;
            PostSendWithRetry(conns[myConnIdx].sendComm, gpuBuf, kWarmupSize,
                              kWarmupTag, mhandles[myConnIdx], &wSendReq);
        }
        if (rank == 0) {
            for (int s = 0; s < numSenders; s++) {
                if (wRecvReqs[s]) {
                    int sz = 0;
                    WaitForCompletion(wRecvReqs[s], &sz, kPCIeTimeoutMs);
                }
            }
        }
        if (wSendReq) {
            int sz = 0;
            WaitForCompletion(wSendReq, &sz, kPCIeTimeoutMs);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "[Warmup] Fan-in connections primed ("
                      << kWarmupSize << " bytes per link)" << std::endl;
    }

    const int iters = getPCIeIters();

    // ── Message size sweep ─────────────────────────────────────────
    for (size_t msgSize : getPCIeMessageSizes()) {
        const size_t segmentSize = msgSize;

        if (rank == 0) {
            printVerifyModeOnce();
            std::cout << "[PCIeOrderingFanIn] msgSize=" << (msgSize / 1024) << "KB"
                      << " senders=" << numSenders
                      << " iters=" << iters << std::endl;
        }

        for (int iter = 0; iter < iters; iter++) {

            // 1. Each sender fills its GPU send buffer with a unique pattern
            if (rank != 0) {
                int seed = rank * 1000 + iter;
                EXPECT_EQ(initializeBufferWithPattern<uint8_t>(
                              gpuBuf, segmentSize, makeBytePattern(seed)),
                          hipSuccess);
            }

            // 2. Rank 0 zeros its full receive buffer
            if (rank == 0) {
                EXPECT_EQ(zeroInitializeBuffer<uint8_t>(gpuBuf, numSenders * segmentSize),
                          hipSuccess);
            }

            // 3. Rank 0 posts N-1 irecvs (one per segment), all at once
            std::vector<void*> recvReqs(numSenders, nullptr);
            if (rank == 0) {
                for (int s = 0; s < numSenders; s++) {
                    char* segPtr = static_cast<char*>(gpuBuf) + s * segmentSize;
                    void*  bufs[1]    = {segPtr};
                    size_t sizes[1]   = {segmentSize};
                    int    tags[1]    = {iter};
                    void*  handles[1] = {mhandles[s]};
                    EXPECT_EQ(PostRecv(conns[s].recvComm, 1, bufs, sizes, tags, handles,
                                      &recvReqs[s]),
                              ncclSuccess);
                }
            }

            // Barrier: ensure all recvs are posted before senders fire
            MPI_Barrier(MPI_COMM_WORLD);

            // 4. All senders begin PostSendWithRetry simultaneously
            void* sendReq = nullptr;
            if (rank != 0) {
                // Find my connection index
                int myConnIdx = rank - 1;
                PostSendWithRetry(conns[myConnIdx].sendComm, gpuBuf, segmentSize,
                                  /*tag=*/iter, mhandles[myConnIdx], &sendReq);
            }

            // 5. Wait for completions
            if (rank == 0) {
                for (int s = 0; s < numSenders; s++) {
                    if (recvReqs[s]) {
                        int sz = 0;
                        EXPECT_EQ(WaitForCompletion(recvReqs[s], &sz, kPCIeTimeoutMs),
                                  ncclSuccess)
                            << "Recv timeout: sender=" << senderRanks[s]
                            << " iter=" << iter;
                    }
                }
            }
            if (sendReq) {
                int sz = 0;
                EXPECT_EQ(WaitForCompletion(sendReq, &sz, kPCIeTimeoutMs), ncclSuccess)
                    << "Send timeout: rank=" << rank << " iter=" << iter;
            }

            // 6. Rank 0 flushes each segment
            if (rank == 0) {
                for (int s = 0; s < numSenders; s++) {
                    char* segPtr = static_cast<char*>(gpuBuf) + s * segmentSize;
                    void* flushBufs[1]    = {segPtr};
                    int   flushSizes[1]   = {static_cast<int>(segmentSize)};
                    void* flushHandles[1] = {mhandles[s]};
                    void* flushReq        = nullptr;
                    ncclResult_t fr = FlushRecv(conns[s].recvComm, 1, flushBufs,
                                               flushSizes, flushHandles, &flushReq);
                    if (fr == ncclSuccess && flushReq) {
                        int fsz = 0;
                        EXPECT_EQ(WaitForCompletion(flushReq, &fsz, kPCIeTimeoutMs),
                                  ncclSuccess);
                    }
                    // flushReq is nullptr when GDR_FLUSH_DISABLE=1 (NIC HW flush only, no SW flush)
                }
            }

            // 7. Rank 0 verifies each segment byte-by-byte
            if (rank == 0) {
                for (int s = 0; s < numSenders; s++) {
                    int senderRk = senderRanks[s];
                    int seed = senderRk * 1000 + iter;
                    char* segPtr = static_cast<char*>(gpuBuf) + s * segmentSize;

                    size_t errIdx = 0;
                    uint8_t errExp = 0, errGot = 0;
                    uint64_t totalErrs = 0;
                    bool ok = verifyBuffer(
                        segPtr, segmentSize, seed,
                        &errIdx, &errExp, &errGot, &totalErrs);
                    EXPECT_TRUE(ok) << "FanIn mismatch: sender=" << senderRk
                                    << " iter=" << iter
                                    << " totalErrors=" << totalErrs
                                    << " firstByteOffset=" << errIdx
                                    << " expected=0x" << std::hex << (int)errExp
                                    << " actual=0x" << (int)errGot << std::dec;
                }
            }

            // Progress logging: ~10 updates total (at 10%, 20%, … 100%)
            if (rank == 0) {
                int step = std::max(1, iters / 10);
                if ((iter + 1) % step == 0 || iter == iters - 1) {
                    std::cout << "  [FanIn " << (msgSize / 1024) << "KB] completed "
                              << (iter + 1) << "/" << iters << " iterations" << std::endl;
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // ── Cleanup ────────────────────────────────────────────────────
    for (size_t i = 0; i < conns.size(); i++) {
        if (!mhandles[i]) continue;
        void* comm = nullptr;
        if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
        if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
        if (comm) DeregisterMemory(comm, mhandles[i]);
        mhandles[i] = nullptr;
    }
    mrCleanup.dismiss();

    for (auto& c : conns) CloseDirectedConnection(c);
}

// =====================================================================
//  Test 2: PCIeOrderingAllGatherRing
//
//  Ring topology: rank k sends to (k+1)%N and receives from (k-1+N)%N.
//  Each rank is both sender and receiver simultaneously, mimicking RCCL
//  ring allgather.
// =====================================================================
TEST_F(NetIbMPITest, PCIeOrderingAllGatherRing)
{
    const int nranks = MPIEnvironment::world_size;
    ASSERT_TRUE(validateTestPrerequisites(
        /*min_processes=*/kMinFourProcesses,
        /*max_processes=*/0,
        /*require_power_of_two=*/false,
        /*min_nodes=*/kMinTwoNodes,
        /*max_nodes=*/kNoNodeLimit))
        << "Test requires at least " << kMinFourProcesses
        << " MPI processes across at least " << kMinTwoNodes << " nodes";

    const int rank = MPIEnvironment::world_rank;

    // Force ROCm IB plugin
    net_ = &rocmNetIb;
    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    // Check GDR support
    ncclNetProperties_t props;
    ASSERT_EQ(GetDeviceProperties(0, &props), ncclSuccess);
    if (!(props.ptrSupport & NCCL_PTR_CUDA)) {
        GTEST_SKIP() << "No GPU Direct RDMA support — skipping PCIe ordering test";
    }

    const int prevRank = (rank - 1 + nranks) % nranks;
    const int nextRank = (rank + 1) % nranks;

    // ── Build ring connections ─────────────────────────────────────
    // N connections: ringConns[s] connects sender=s → receiver=(s+1)%N
    std::vector<DirectedConnection> ringConns(nranks);
    for (int s = 0; s < nranks; s++) {
        SetupDirectedConnection(/*dev=*/0, ringConns[s],
                                /*senderRank=*/s,
                                /*receiverRank=*/(s + 1) % nranks,
                                /*mpiTag=*/400 + s);
    }

    // Find my send and recv connections
    DirectedConnection& mySendConn = ringConns[rank];
    DirectedConnection& myRecvConn = ringConns[prevRank];

    const size_t maxMsgSize = kAllMessageSizes.back();

    // ── GPU buffer allocation ──────────────────────────────────────
    // Send buffer: my own data
    void* gpuSendBuf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&gpuSendBuf, maxMsgSize));
    auto sendGuard = makeDeviceBufferAutoGuard(gpuSendBuf);

    // Recv buffer: data from prevRank
    void* gpuRecvBuf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&gpuRecvBuf, maxMsgSize));
    auto recvGuard = makeDeviceBufferAutoGuard(gpuRecvBuf);

    // ── Register MRs ───────────────────────────────────────────────
    void* sendMh = nullptr;
    void* recvMh = nullptr;
    ASSERT_EQ(RegisterMemory(mySendConn.sendComm, gpuSendBuf, maxMsgSize,
                             NCCL_PTR_CUDA, &sendMh),
              ncclSuccess);
    ASSERT_EQ(RegisterMemory(myRecvConn.recvComm, gpuRecvBuf, maxMsgSize,
                             NCCL_PTR_CUDA, &recvMh),
              ncclSuccess);

    auto mrCleanup = makeScopeGuard([&]() {
        if (sendMh) DeregisterMemory(mySendConn.sendComm, sendMh);
        if (recvMh) DeregisterMemory(myRecvConn.recvComm, recvMh);
    });

    // ── Warm up: prime all ring connections ────────────────────────
    {
        MPI_Barrier(MPI_COMM_WORLD);
        void* wRecvReq = nullptr;
        {
            void*  bufs[1]    = {gpuRecvBuf};
            size_t sizes[1]   = {kWarmupSize};
            int    tags[1]    = {kWarmupTag};
            void*  handles[1] = {recvMh};
            PostRecv(myRecvConn.recvComm, 1, bufs, sizes, tags, handles, &wRecvReq);
        }
        void* wSendReq = nullptr;
        PostSendWithRetry(mySendConn.sendComm, gpuSendBuf, kWarmupSize,
                          kWarmupTag, sendMh, &wSendReq);
        if (wSendReq) { int sz = 0; WaitForCompletion(wSendReq, &sz, kPCIeTimeoutMs); }
        if (wRecvReq) { int sz = 0; WaitForCompletion(wRecvReq, &sz, kPCIeTimeoutMs); }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "[Warmup] Ring connections primed ("
                      << kWarmupSize << " bytes per link)" << std::endl;
    }

    const int iters = getPCIeIters();

    // ── Message size sweep ─────────────────────────────────────────
    for (size_t msgSize : getPCIeMessageSizes()) {

        if (rank == 0) {
            std::cout << "[PCIeOrderingRing] msgSize=" << (msgSize / 1024) << "KB"
                      << " nranks=" << nranks
                      << " iters=" << iters << std::endl;
        }

        for (int iter = 0; iter < iters; iter++) {

            // 1. Each rank fills its send buffer with its own pattern
            int sendSeed = rank * 1000 + iter;
            EXPECT_EQ(initializeBufferWithPattern<uint8_t>(
                          gpuSendBuf, msgSize, makeBytePattern(sendSeed)),
                      hipSuccess);

            // 2. Each rank zeros the recv buffer
            EXPECT_EQ(zeroInitializeBuffer<uint8_t>(gpuRecvBuf, msgSize), hipSuccess);

            // 3. Barrier: ensure all buffers are prepared
            MPI_Barrier(MPI_COMM_WORLD);

            // 4. Each rank posts irecv from prevRank AND isend to nextRank
            void* recvReq = nullptr;
            {
                void*  bufs[1]    = {gpuRecvBuf};
                size_t sizes[1]   = {msgSize};
                int    tags[1]    = {iter};
                void*  handles[1] = {recvMh};
                EXPECT_EQ(PostRecv(myRecvConn.recvComm, 1, bufs, sizes, tags,
                                   handles, &recvReq),
                          ncclSuccess);
            }

            void* sendReq = nullptr;
            PostSendWithRetry(mySendConn.sendComm, gpuSendBuf, msgSize,
                              /*tag=*/iter, sendMh, &sendReq);

            // 5. Wait for both send and recv completion
            int sendComplSize = 0, recvComplSize = 0;
            ncclResult_t sendResult = ncclSuccess, recvResult = ncclSuccess;

            if (sendReq) {
                sendResult = WaitForCompletion(sendReq, &sendComplSize, kPCIeTimeoutMs);
                EXPECT_EQ(sendResult, ncclSuccess)
                    << "Ring send timeout: rank=" << rank << " iter=" << iter;
            }
            if (recvReq) {
                recvResult = WaitForCompletion(recvReq, &recvComplSize, kPCIeTimeoutMs);
                EXPECT_EQ(recvResult, ncclSuccess)
                    << "Ring recv timeout: rank=" << rank << " iter=" << iter;
            }

            // 6. Flush the received segment
            ncclResult_t flushResult = ncclSuccess;
            {
                void* flushBufs[1]    = {gpuRecvBuf};
                int   flushSizes[1]   = {static_cast<int>(msgSize)};
                void* flushHandles[1] = {recvMh};
                void* flushReq        = nullptr;
                ncclResult_t fr = FlushRecv(myRecvConn.recvComm, 1, flushBufs,
                                            flushSizes, flushHandles, &flushReq);
                if (fr == ncclSuccess && flushReq) {
                    int fsz = 0;
                    flushResult = WaitForCompletion(flushReq, &fsz, kPCIeTimeoutMs);
                    EXPECT_EQ(flushResult, ncclSuccess);
                }
                // flushReq is nullptr when GDR_FLUSH_DISABLE=1 (NIC HW flush only, no SW flush)
            }

            // 7. Each rank verifies the received data
            {
                int expectedSeed = prevRank * 1000 + iter;
                size_t errIdx = 0;
                uint8_t errExp = 0, errGot = 0;
                uint64_t totalErrs = 0;
                bool ok = verifyBuffer(
                    gpuRecvBuf, msgSize, expectedSeed,
                    &errIdx, &errExp, &errGot, &totalErrs);

                if (!ok) {
                    std::cout << "  [DIAG rank=" << rank
                              << " iter=" << iter << "] "
                              << "send=" << (sendResult == ncclSuccess ? "OK" : "FAIL")
                              << "(sz=" << sendComplSize << ")"
                              << " recv=" << (recvResult == ncclSuccess ? "OK" : "FAIL")
                              << "(sz=" << recvComplSize << "/" << msgSize << ")"
                              << " flush=" << (flushResult == ncclSuccess ? "OK" : "FAIL")
                              << " from=rank" << prevRank
                              << " isWraparound=" << (prevRank == nranks - 1 ? "YES" : "no")
                              << std::endl;
                }

                EXPECT_TRUE(ok) << "Ring mismatch: rank=" << rank
                                << " fromRank=" << prevRank
                                << " iter=" << iter
                                << " totalErrors=" << totalErrs
                                << " firstByteOffset=" << errIdx
                                << " expected=0x" << std::hex << (int)errExp
                                << " actual=0x" << (int)errGot << std::dec;
            }

            // Progress logging: ~10 updates total (at 10%, 20%, … 100%)
            if (rank == 0) {
                int step = std::max(1, iters / 10);
                if ((iter + 1) % step == 0 || iter == iters - 1) {
                    std::cout << "  [Ring " << (msgSize / 1024) << "KB] completed "
                              << (iter + 1) << "/" << iters << " iterations" << std::endl;
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // ── Cleanup ────────────────────────────────────────────────────
    if (sendMh) {
        DeregisterMemory(mySendConn.sendComm, sendMh);
        sendMh = nullptr;
    }
    if (recvMh) {
        DeregisterMemory(myRecvConn.recvComm, recvMh);
        recvMh = nullptr;
    }
    mrCleanup.dismiss();

    for (auto& c : ringConns) CloseDirectedConnection(c);
}

// =====================================================================
//  Test 3: PCIeOrderingMultiNICFanIn
//
//  Multi-NIC variant of fan-in test. Run with -npernode N (N >= 2) so
//  each MPI rank binds to a different GPU and uses a different NIC
//  (dev = local_rank % ndev). Stresses multiple PCIe paths per node.
//
//  Example: mpirun -np 16 -npernode 8 -H nodeA,nodeB ...
// =====================================================================
TEST_F(NetIbMPITest, PCIeOrderingMultiNICFanIn)
{
    const int nranks = MPIEnvironment::world_size;
    ASSERT_TRUE(validateTestPrerequisites(
        /*min_processes=*/kMinFourProcesses,
        /*max_processes=*/0,
        /*require_power_of_two=*/false,
        /*min_nodes=*/kMinTwoNodes,
        /*max_nodes=*/kNoNodeLimit))
        << "Test requires at least " << kMinFourProcesses
        << " MPI processes across at least " << kMinTwoNodes << " nodes";

    const int rank = MPIEnvironment::world_rank;
    const int localRank = getLocalRank();

    // Force ROCm IB plugin
    net_ = &rocmNetIb;
    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    // Each rank uses its own NIC device based on local_rank
    const int myDev = localRank % ndev;

    // Check GDR support on assigned device
    ncclNetProperties_t props;
    ASSERT_EQ(GetDeviceProperties(myDev, &props), ncclSuccess);
    if (!(props.ptrSupport & NCCL_PTR_CUDA)) {
        GTEST_SKIP() << "No GPU Direct RDMA support on dev " << myDev;
    }

    // Build rank→device map for all ranks (gathered via MPI)
    std::vector<int> rankDevMap(nranks);
    MPI_Allgather(&myDev, 1, MPI_INT, rankDevMap.data(), 1, MPI_INT, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "[PCIeOrderingMultiNICFanIn] nranks=" << nranks
                  << " ndev=" << ndev << " rank→dev:";
        for (int r = 0; r < nranks; r++)
            std::cout << " " << r << "→" << rankDevMap[r];
        std::cout << std::endl;
    }

    // Build sender list: all ranks except rank 0
    std::vector<int> senderRanks;
    for (int r = 1; r < nranks; r++)
        senderRanks.push_back(r);

    // Setup fan-in using per-rank device mapping
    std::vector<DirectedConnection> conns(senderRanks.size());
    for (size_t i = 0; i < senderRanks.size(); i++) {
        SetupDirectedConnectionMultiDev(rankDevMap, conns[i],
                                        senderRanks[i], /*receiverRank=*/0,
                                        /*mpiTag=*/500 + static_cast<int>(i));
    }

    const int numSenders = static_cast<int>(senderRanks.size());
    const size_t maxMsgSize = kAllMessageSizes.back();
    const size_t maxTotalRecvSize = maxMsgSize * numSenders;

    // GPU buffer allocation
    void* gpuBuf = nullptr;
    size_t gpuBufSize = (rank == 0) ? maxTotalRecvSize : maxMsgSize;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&gpuBuf, gpuBufSize));
    auto gpuGuard = makeDeviceBufferAutoGuard(gpuBuf);

    // Register MR once at max size
    std::vector<void*> mhandles(conns.size(), nullptr);
    for (size_t i = 0; i < conns.size(); i++) {
        void* comm = nullptr;
        if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
        if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
        if (comm) {
            ASSERT_EQ(RegisterMemory(comm, gpuBuf, gpuBufSize, NCCL_PTR_CUDA, &mhandles[i]),
                      ncclSuccess);
        }
    }
    auto mrCleanup = makeScopeGuard([&]() {
        for (size_t i = 0; i < conns.size(); i++) {
            if (!mhandles[i]) continue;
            void* comm = nullptr;
            if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
            if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
            if (comm) DeregisterMemory(comm, mhandles[i]);
        }
    });

    // ── Warm up: prime all fan-in connections ────────────────────
    {
        MPI_Barrier(MPI_COMM_WORLD);
        std::vector<void*> wRecvReqs(numSenders, nullptr);
        if (rank == 0) {
            for (int s = 0; s < numSenders; s++) {
                char* segPtr = static_cast<char*>(gpuBuf) + s * kWarmupSize;
                void*  bufs[1]    = {segPtr};
                size_t sizes[1]   = {kWarmupSize};
                int    tags[1]    = {kWarmupTag};
                void*  handles[1] = {mhandles[s]};
                PostRecv(conns[s].recvComm, 1, bufs, sizes, tags, handles,
                         &wRecvReqs[s]);
            }
        }
        void* wSendReq = nullptr;
        if (rank != 0) {
            int myConnIdx = rank - 1;
            PostSendWithRetry(conns[myConnIdx].sendComm, gpuBuf, kWarmupSize,
                              kWarmupTag, mhandles[myConnIdx], &wSendReq);
        }
        if (rank == 0) {
            for (int s = 0; s < numSenders; s++) {
                if (wRecvReqs[s]) {
                    int sz = 0;
                    WaitForCompletion(wRecvReqs[s], &sz, kPCIeTimeoutMs);
                }
            }
        }
        if (wSendReq) {
            int sz = 0;
            WaitForCompletion(wSendReq, &sz, kPCIeTimeoutMs);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "[Warmup] Fan-in connections primed ("
                      << kWarmupSize << " bytes per link)" << std::endl;
    }

    const int iters = getPCIeIters();

    for (size_t msgSize : getPCIeMessageSizes()) {
        const size_t segmentSize = msgSize;

        if (rank == 0) {
            std::cout << "[MultiNICFanIn] msgSize=" << (msgSize / 1024) << "KB"
                      << " senders=" << numSenders
                      << " iters=" << iters << std::endl;
        }

        for (int iter = 0; iter < iters; iter++) {
            if (rank != 0) {
                int seed = rank * 1000 + iter;
                EXPECT_EQ(initializeBufferWithPattern<uint8_t>(
                              gpuBuf, segmentSize, makeBytePattern(seed)),
                          hipSuccess);
            }

            if (rank == 0) {
                EXPECT_EQ(zeroInitializeBuffer<uint8_t>(gpuBuf, numSenders * segmentSize),
                          hipSuccess);
            }

            std::vector<void*> recvReqs(numSenders, nullptr);
            if (rank == 0) {
                for (int s = 0; s < numSenders; s++) {
                    char* segPtr = static_cast<char*>(gpuBuf) + s * segmentSize;
                    void*  bufs[1]    = {segPtr};
                    size_t sizes[1]   = {segmentSize};
                    int    tags[1]    = {iter};
                    void*  handles[1] = {mhandles[s]};
                    EXPECT_EQ(PostRecv(conns[s].recvComm, 1, bufs, sizes, tags, handles,
                                      &recvReqs[s]),
                              ncclSuccess);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);

            void* sendReq = nullptr;
            if (rank != 0) {
                int myConnIdx = rank - 1;
                PostSendWithRetry(conns[myConnIdx].sendComm, gpuBuf, segmentSize,
                                  /*tag=*/iter, mhandles[myConnIdx], &sendReq);
            }

            if (rank == 0) {
                for (int s = 0; s < numSenders; s++) {
                    if (recvReqs[s]) {
                        int sz = 0;
                        EXPECT_EQ(WaitForCompletion(recvReqs[s], &sz, kPCIeTimeoutMs),
                                  ncclSuccess)
                            << "Recv timeout: sender=" << senderRanks[s]
                            << " iter=" << iter;
                    }
                }
            }
            if (sendReq) {
                int sz = 0;
                EXPECT_EQ(WaitForCompletion(sendReq, &sz, kPCIeTimeoutMs), ncclSuccess)
                    << "Send timeout: rank=" << rank << " dev=" << myDev << " iter=" << iter;
            }

            if (rank == 0) {
                for (int s = 0; s < numSenders; s++) {
                    char* segPtr = static_cast<char*>(gpuBuf) + s * segmentSize;
                    void* flushBufs[1]    = {segPtr};
                    int   flushSizes[1]   = {static_cast<int>(segmentSize)};
                    void* flushHandles[1] = {mhandles[s]};
                    void* flushReq        = nullptr;
                    ncclResult_t fr = FlushRecv(conns[s].recvComm, 1, flushBufs,
                                               flushSizes, flushHandles, &flushReq);
                    if (fr == ncclSuccess && flushReq) {
                        int fsz = 0;
                        EXPECT_EQ(WaitForCompletion(flushReq, &fsz, kPCIeTimeoutMs),
                                  ncclSuccess);
                    }
                }
            }

            if (rank == 0) {
                for (int s = 0; s < numSenders; s++) {
                    int senderRk = senderRanks[s];
                    int seed = senderRk * 1000 + iter;
                    char* segPtr = static_cast<char*>(gpuBuf) + s * segmentSize;

                    size_t errIdx = 0;
                    uint8_t errExp = 0, errGot = 0;
                    uint64_t totalErrs = 0;
                    bool ok = verifyBuffer(
                        segPtr, segmentSize, seed,
                        &errIdx, &errExp, &errGot, &totalErrs);
                    EXPECT_TRUE(ok) << "MultiNICFanIn mismatch: sender=" << senderRk
                                    << " dev=" << rankDevMap[senderRk]
                                    << " iter=" << iter
                                    << " totalErrors=" << totalErrs
                                    << " firstByteOffset=" << errIdx
                                    << " expected=0x" << std::hex << (int)errExp
                                    << " actual=0x" << (int)errGot << std::dec;
                }
            }

            if (rank == 0) {
                int step = std::max(1, iters / 10);
                if ((iter + 1) % step == 0 || iter == iters - 1) {
                    std::cout << "  [MultiNICFanIn " << (msgSize / 1024) << "KB] completed "
                              << (iter + 1) << "/" << iters << " iterations" << std::endl;
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // Cleanup
    for (size_t i = 0; i < conns.size(); i++) {
        if (!mhandles[i]) continue;
        void* comm = nullptr;
        if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
        if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
        if (comm) DeregisterMemory(comm, mhandles[i]);
        mhandles[i] = nullptr;
    }
    mrCleanup.dismiss();

    for (auto& c : conns) CloseDirectedConnection(c);
}

// =====================================================================
//  Test 4: PCIeOrderingMultiNICRing
//
//  Multi-NIC variant of ring test. Run with -npernode N (N >= 2) so
//  each MPI rank uses a different NIC. Ring topology spans across all
//  ranks and NICs.
//
//  Example: mpirun -np 16 -npernode 8 -H nodeA,nodeB ...
// =====================================================================
TEST_F(NetIbMPITest, PCIeOrderingMultiNICRing)
{
    const int nranks = MPIEnvironment::world_size;
    ASSERT_TRUE(validateTestPrerequisites(
        /*min_processes=*/kMinFourProcesses,
        /*max_processes=*/0,
        /*require_power_of_two=*/false,
        /*min_nodes=*/kMinTwoNodes,
        /*max_nodes=*/kNoNodeLimit))
        << "Test requires at least " << kMinFourProcesses
        << " MPI processes across at least " << kMinTwoNodes << " nodes";

    const int rank = MPIEnvironment::world_rank;
    const int localRank = getLocalRank();

    // Force ROCm IB plugin
    net_ = &rocmNetIb;
    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    const int myDev = localRank % ndev;

    // Check GDR support on assigned device
    ncclNetProperties_t props;
    ASSERT_EQ(GetDeviceProperties(myDev, &props), ncclSuccess);
    if (!(props.ptrSupport & NCCL_PTR_CUDA)) {
        GTEST_SKIP() << "No GPU Direct RDMA support on dev " << myDev;
    }

    // Build rank→device map
    std::vector<int> rankDevMap(nranks);
    MPI_Allgather(&myDev, 1, MPI_INT, rankDevMap.data(), 1, MPI_INT, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "[PCIeOrderingMultiNICRing] nranks=" << nranks
                  << " ndev=" << ndev << " rank→dev:";
        for (int r = 0; r < nranks; r++)
            std::cout << " " << r << "→" << rankDevMap[r];
        std::cout << std::endl;
    }

    const int prevRank = (rank - 1 + nranks) % nranks;

    // Build ring connections using per-rank device mapping
    // Setup order: start from the wraparound connection (nranks-1 → 0)
    // then proceed forward. This diagnoses whether failures correlate
    // with "last-created connection" vs a specific link.
    std::vector<DirectedConnection> ringConns(nranks);
    {
        // DIAGNOSTIC: sequential order (0→1 first, wraparound last) to reproduce bug
        std::vector<int> setupOrder;
        for (int s = 0; s < nranks; s++)
            setupOrder.push_back(s);

        if (rank == 0) {
            std::cout << "[MultiNICRing] Connection setup order:";
            for (int s : setupOrder)
                std::cout << " " << s << "→" << (s + 1) % nranks;
            std::cout << std::endl;
        }

        for (int s : setupOrder) {
            SetupDirectedConnectionMultiDev(rankDevMap, ringConns[s],
                                            /*senderRank=*/s,
                                            /*receiverRank=*/(s + 1) % nranks,
                                            /*mpiTag=*/600 + s);
        }
    }

    DirectedConnection& mySendConn = ringConns[rank];
    DirectedConnection& myRecvConn = ringConns[prevRank];

    const size_t maxMsgSize = kAllMessageSizes.back();

    // GPU buffer allocation
    void* gpuSendBuf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&gpuSendBuf, maxMsgSize));
    auto sendGuard = makeDeviceBufferAutoGuard(gpuSendBuf);

    void* gpuRecvBuf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&gpuRecvBuf, maxMsgSize));
    auto recvGuard = makeDeviceBufferAutoGuard(gpuRecvBuf);

    // Register MRs
    void* sendMh = nullptr;
    void* recvMh = nullptr;
    ASSERT_EQ(RegisterMemory(mySendConn.sendComm, gpuSendBuf, maxMsgSize,
                             NCCL_PTR_CUDA, &sendMh),
              ncclSuccess);
    ASSERT_EQ(RegisterMemory(myRecvConn.recvComm, gpuRecvBuf, maxMsgSize,
                             NCCL_PTR_CUDA, &recvMh),
              ncclSuccess);

    auto mrCleanup = makeScopeGuard([&]() {
        if (sendMh) DeregisterMemory(mySendConn.sendComm, sendMh);
        if (recvMh) DeregisterMemory(myRecvConn.recvComm, recvMh);
    });

    // ── Warm up: prime all ring connections ────────────────────────
    {
        MPI_Barrier(MPI_COMM_WORLD);
        void* wRecvReq = nullptr;
        {
            void*  bufs[1]    = {gpuRecvBuf};
            size_t sizes[1]   = {kWarmupSize};
            int    tags[1]    = {kWarmupTag};
            void*  handles[1] = {recvMh};
            PostRecv(myRecvConn.recvComm, 1, bufs, sizes, tags, handles, &wRecvReq);
        }
        void* wSendReq = nullptr;
        PostSendWithRetry(mySendConn.sendComm, gpuSendBuf, kWarmupSize,
                          kWarmupTag, sendMh, &wSendReq);
        if (wSendReq) { int sz = 0; WaitForCompletion(wSendReq, &sz, kPCIeTimeoutMs); }
        if (wRecvReq) { int sz = 0; WaitForCompletion(wRecvReq, &sz, kPCIeTimeoutMs); }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "[Warmup] Ring connections primed ("
                      << kWarmupSize << " bytes per link)" << std::endl;
    }

    // ── Point-to-point isolation test: verify each link individually ──
    // Serializes sends so only ONE link is active at a time.
    // If a link fails here, the connection itself is broken.
    // If it passes here but fails in the ring, concurrent traffic is the trigger.
    {
        const size_t isolationSize = 4 * 1024 * 1024;  // 4 MB — matches failure size
        if (rank == 0)
            std::cout << "[IsolationTest] Testing each ring link individually ("
                      << (isolationSize / 1024) << " KB)..." << std::endl;

        for (int link = 0; link < nranks; link++) {
            int sender   = link;
            int receiver = (link + 1) % nranks;

            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == receiver) {
                // Initialize recv buffer to zeros
                EXPECT_EQ(zeroInitializeBuffer<uint8_t>(gpuRecvBuf, isolationSize),
                          hipSuccess);

                int seed = sender * 1000 + 777;  // deterministic seed

                void* recvReq = nullptr;
                {
                    void*  bufs[1]    = {gpuRecvBuf};
                    size_t sizes[1]   = {isolationSize};
                    int    tags[1]    = {kWarmupTag + 1 + link};
                    void*  handles[1] = {recvMh};
                    PostRecv(myRecvConn.recvComm, 1, bufs, sizes, tags,
                             handles, &recvReq);
                }

                if (recvReq) {
                    int rsz = 0;
                    ncclResult_t rr = WaitForCompletion(recvReq, &rsz, kPCIeTimeoutMs);
                    EXPECT_EQ(rr, ncclSuccess)
                        << "[IsolationTest] recv timeout link " << sender << "→" << receiver;

                    // Flush
                    void* flushBufs[1]    = {gpuRecvBuf};
                    int   flushSizes[1]   = {static_cast<int>(isolationSize)};
                    void* flushHandles[1] = {recvMh};
                    void* flushReq        = nullptr;
                    ncclResult_t fr = FlushRecv(myRecvConn.recvComm, 1, flushBufs,
                                                flushSizes, flushHandles, &flushReq);
                    if (fr == ncclSuccess && flushReq) {
                        int fsz = 0;
                        WaitForCompletion(flushReq, &fsz, kPCIeTimeoutMs);
                    }

                    // Verify
                    size_t errIdx = 0;
                    uint8_t errExp = 0, errGot = 0;
                    uint64_t totalErrs = 0;
                    bool ok = verifyBuffer(gpuRecvBuf, isolationSize, seed,
                                           &errIdx, &errExp, &errGot, &totalErrs);
                    std::cout << "[IsolationTest] Link " << sender << "→" << receiver
                              << " (dev" << rankDevMap[sender] << "→dev" << rankDevMap[receiver]
                              << "): " << (ok ? "PASS" : "FAIL");
                    if (!ok) {
                        std::cout << " firstErr@" << errIdx
                                  << " exp=0x" << std::hex << (int)errExp
                                  << " got=0x" << (int)errGot << std::dec
                                  << " totalErrs=" << totalErrs;
                    }
                    std::cout << std::endl;
                    EXPECT_TRUE(ok) << "[IsolationTest] Link " << sender
                                    << "→" << receiver << " FAILED";
                }
            }
            else if (rank == sender) {
                int seed = sender * 1000 + 777;
                EXPECT_EQ(initializeBufferWithPattern<uint8_t>(
                              gpuSendBuf, isolationSize, makeBytePattern(seed)),
                          hipSuccess);

                void* sendReq = nullptr;
                PostSendWithRetry(mySendConn.sendComm, gpuSendBuf, isolationSize,
                                  kWarmupTag + 1 + link, sendMh, &sendReq);
                if (sendReq) {
                    int ssz = 0;
                    ncclResult_t sr = WaitForCompletion(sendReq, &ssz, kPCIeTimeoutMs);
                    EXPECT_EQ(sr, ncclSuccess)
                        << "[IsolationTest] send timeout link " << sender << "→" << receiver;
                }
            }
            // Other ranks just barrier
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "[IsolationTest] All links tested individually." << std::endl;
    }

    const int iters = getPCIeIters();

    for (size_t msgSize : getPCIeMessageSizes()) {

        if (rank == 0) {
            std::cout << "[MultiNICRing] msgSize=" << (msgSize / 1024) << "KB"
                      << " nranks=" << nranks
                      << " iters=" << iters << std::endl;
        }

        for (int iter = 0; iter < iters; iter++) {
            int sendSeed = rank * 1000 + iter;
            EXPECT_EQ(initializeBufferWithPattern<uint8_t>(
                          gpuSendBuf, msgSize, makeBytePattern(sendSeed)),
                      hipSuccess);

            EXPECT_EQ(zeroInitializeBuffer<uint8_t>(gpuRecvBuf, msgSize), hipSuccess);

            MPI_Barrier(MPI_COMM_WORLD);

            void* recvReq = nullptr;
            {
                void*  bufs[1]    = {gpuRecvBuf};
                size_t sizes[1]   = {msgSize};
                int    tags[1]    = {iter};
                void*  handles[1] = {recvMh};
                EXPECT_EQ(PostRecv(myRecvConn.recvComm, 1, bufs, sizes, tags,
                                   handles, &recvReq),
                          ncclSuccess);
            }

            void* sendReq = nullptr;
            PostSendWithRetry(mySendConn.sendComm, gpuSendBuf, msgSize,
                              /*tag=*/iter, sendMh, &sendReq);

            // Track completion status for diagnostic logging
            int sendComplSize = 0, recvComplSize = 0;
            ncclResult_t sendResult = ncclSuccess, recvResult = ncclSuccess;

            if (sendReq) {
                sendResult = WaitForCompletion(sendReq, &sendComplSize, kPCIeTimeoutMs);
                EXPECT_EQ(sendResult, ncclSuccess)
                    << "Ring send timeout: rank=" << rank << " dev=" << myDev << " iter=" << iter;
            }
            if (recvReq) {
                recvResult = WaitForCompletion(recvReq, &recvComplSize, kPCIeTimeoutMs);
                EXPECT_EQ(recvResult, ncclSuccess)
                    << "Ring recv timeout: rank=" << rank << " dev=" << myDev << " iter=" << iter;
            }

            ncclResult_t flushResult = ncclSuccess;
            {
                void* flushBufs[1]    = {gpuRecvBuf};
                int   flushSizes[1]   = {static_cast<int>(msgSize)};
                void* flushHandles[1] = {recvMh};
                void* flushReq        = nullptr;
                ncclResult_t fr = FlushRecv(myRecvConn.recvComm, 1, flushBufs,
                                            flushSizes, flushHandles, &flushReq);
                if (fr == ncclSuccess && flushReq) {
                    int fsz = 0;
                    flushResult = WaitForCompletion(flushReq, &fsz, kPCIeTimeoutMs);
                    EXPECT_EQ(flushResult, ncclSuccess);
                }
            }

            {
                int expectedSeed = prevRank * 1000 + iter;
                size_t errIdx = 0;
                uint8_t errExp = 0, errGot = 0;
                uint64_t totalErrs = 0;
                bool ok = verifyBuffer(
                    gpuRecvBuf, msgSize, expectedSeed,
                    &errIdx, &errExp, &errGot, &totalErrs);

                if (!ok) {
                    // Per-rank diagnostic: show completion status for this rank
                    std::cout << "  [DIAG rank=" << rank << " dev=" << myDev
                              << " iter=" << iter << "] "
                              << "send=" << (sendResult == ncclSuccess ? "OK" : "FAIL")
                              << "(sz=" << sendComplSize << ")"
                              << " recv=" << (recvResult == ncclSuccess ? "OK" : "FAIL")
                              << "(sz=" << recvComplSize << "/" << msgSize << ")"
                              << " flush=" << (flushResult == ncclSuccess ? "OK" : "FAIL")
                              << " from=rank" << prevRank << "/dev" << rankDevMap[prevRank]
                              << " isWraparound=" << (prevRank == nranks - 1 ? "YES" : "no")
                              << std::endl;
                }

                EXPECT_TRUE(ok) << "MultiNICRing mismatch: rank=" << rank
                                << " dev=" << myDev
                                << " fromRank=" << prevRank
                                << " fromDev=" << rankDevMap[prevRank]
                                << " iter=" << iter
                                << " totalErrors=" << totalErrs
                                << " firstByteOffset=" << errIdx
                                << " expected=0x" << std::hex << (int)errExp
                                << " actual=0x" << (int)errGot << std::dec;
            }

            if (rank == 0) {
                int step = std::max(1, iters / 10);
                if ((iter + 1) % step == 0 || iter == iters - 1) {
                    std::cout << "  [MultiNICRing " << (msgSize / 1024) << "KB] completed "
                              << (iter + 1) << "/" << iters << " iterations" << std::endl;
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // Cleanup
    if (sendMh) {
        DeregisterMemory(mySendConn.sendComm, sendMh);
        sendMh = nullptr;
    }
    if (recvMh) {
        DeregisterMemory(myRecvConn.recvComm, recvMh);
        recvMh = nullptr;
    }
    mrCleanup.dismiss();

    for (auto& c : ringConns) CloseDirectedConnection(c);
}

#endif // MPI_TESTS_ENABLED
