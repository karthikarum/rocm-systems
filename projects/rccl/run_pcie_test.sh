#!/bin/bash
# =============================================================================
# PCIe Ordering Stress Test Runner
# =============================================================================
# Execute on a compute node. Supports single-NIC and multi-NIC scenarios.
#
# Verification: All tests use GPU CU-based verification — a HIP kernel reads
# VRAM via the CU load path (L2 → VRAM), NOT the DMA engine (hipMemcpy).
# This exercises the exact read path a real GPU kernel would use after NIC
# RDMA writes land in VRAM, catching PCIe coherency issues that DMA-based
# checks would miss.
#
# Usage: bash run_pcie_test.sh
#   All parameters are configured below — edit the CONFIGURATION section.
# =============================================================================

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
# Edit these parameters to change test behavior. No command-line args needed.

# Test mode: "single" (1 NIC/node) or "multi" (multiple NICs/node)
MODE="multi"

# Number of NICs/GPUs per node (only used when MODE="multi", range: 2-8)
NICS_PER_NODE=4

# Test filter: which tests to run
#   "*PCIeOrdering*"                  — all 4 tests (single + multi-NIC)
#   "*PCIeOrderingAllGather*"         — single-NIC tests only (FanIn + Ring)
#   "*MultiNIC*"                      — multi-NIC tests only (FanIn + Ring)
#   "*PCIeOrderingAllGatherFanIn"     — single-NIC fan-in only
#   "*PCIeOrderingAllGatherRing"      — single-NIC ring only
#   "*PCIeOrderingMultiNICFanIn"      — multi-NIC fan-in only
#   "*PCIeOrderingMultiNICRing"       — multi-NIC ring only
TEST_FILTER="*PCIeOrdering*"

# Iterations per message size (default: 200 if unset)
ITERS=10

# GDR flush mode:
#   Mode 1 (NIC HW flush only — validates NIC HW flush correctness):
#     GDR_FLUSH_DISABLE=1, GDR_FLUSH_NO_RO=0
#     No explicit SW flush from CPU. Relies entirely on NIC HW to ensure
#     PCIe write ordering before CQE. Failures here indicate a NIC HW flush bug.
#   Mode 2 (SW flush: single RDMA READ from data buffer):
#     GDR_FLUSH_DISABLE=0, GDR_FLUSH_NO_RO=0
#     CPU-side explicit flush fallback.
#   Mode 3 (SW flush: WRITE+READ to dedicated GPU dummy buffer):
#     GDR_FLUSH_DISABLE=0, GDR_FLUSH_NO_RO=1
#     Another CPU-side explicit flush variant.
GDR_FLUSH_DISABLE=1
GDR_FLUSH_NO_RO=0

# GPU CU verification: 1 = GPU kernel reads VRAM via CU load path (default)
#                      0 = DMA-based (hipMemcpy to host, CPU compare)
# GPU CU mode catches PCIe coherency issues; DMA mode is a baseline check.
GPU_VERIFY=1

# Message size range (supports KB/MB suffix, e.g., "64KB", "4MB", or raw bytes "65536")
# Available sizes: 64KB, 256KB, 1MB, 4MB, 16MB
# Only sizes within [MSG_MIN, MSG_MAX] are tested.
#MSG_MIN="64KB"
#MSG_MAX="16MB"
MSG_MIN="4MB"
MSG_MAX="256MB"

# Node list (space-separated)
NODE_LIST=(
    smc300x-ccs-aus-gpud334
    smc300x-ccs-aus-gpuf2ba
)
    #smc300x-ccs-aus-gpuf2ae
    #smc300x-ccs-aus-gpufccf

# ─── END CONFIGURATION ─────────────────────────────────────────────────────

# --- Environment setup (do not edit) ---
export MPI_PATH=/apps/shared/karthik/repo/ompi/install
export PATH=${MPI_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MPI_PATH}/lib:${LD_LIBRARY_PATH}

RCCL_HOME=/apps/shared/karthik/repo/integration/rccl-rocm-sys/rocm-systems/projects/rccl
cd ${RCCL_HOME}/build/debug/test

NUM_NODES=${#NODE_LIST[@]}

# --- Compute ranks and host string based on mode ---
if [ "${MODE}" = "multi" ]; then
    RANKS_PER_NODE=${NICS_PER_NODE}
    # Auto-select multi-NIC tests if user left default filter
    if [ "${TEST_FILTER}" = "*PCIeOrdering*" ]; then
        TEST_FILTER="*MultiNIC*"
    fi
else
    RANKS_PER_NODE=1
fi

NP=$((RANKS_PER_NODE * NUM_NODES))

# Build host string with slot counts
HOSTS=""
for node in "${NODE_LIST[@]}"; do
    [ -n "${HOSTS}" ] && HOSTS="${HOSTS},"
    HOSTS="${HOSTS}${node}:${RANKS_PER_NODE}"
done

# --- Print configuration ---
echo "=== PCIe Ordering Stress Test ==="
echo "Mode:            ${MODE}"
echo "Nodes:           ${NUM_NODES} (${NODE_LIST[*]})"
echo "Ranks per node:  ${RANKS_PER_NODE}"
echo "Total ranks:     ${NP}"
echo "Iterations:      ${ITERS}"
echo "Message range:   ${MSG_MIN} — ${MSG_MAX}"
echo "Flush mode:      GDR_FLUSH_DISABLE=${GDR_FLUSH_DISABLE}, NO_RO=${GDR_FLUSH_NO_RO}"
echo "Test filter:     ${TEST_FILTER}"
if [ "${GPU_VERIFY}" = "1" ]; then
    echo "Verification:    GPU CU kernel (CU load path)"
else
    echo "Verification:    DMA (hipMemcpy to host)"
fi
echo "=================================="

# --- Run ---
mpirun --prefix ${MPI_PATH} \
  -np ${NP} -H ${HOSTS} -npernode ${RANKS_PER_NODE} --bind-to numa \
  --mca btl ^openib \
  --mca btl_tcp_if_include ens50f0 \
  --mca orte_base_help_aggregate 0 \
  -x NCCL_GDR_FLUSH_DISABLE=${GDR_FLUSH_DISABLE} \
  -x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=${GDR_FLUSH_NO_RO} \
  -x PCIE_ORDER_ITERS=${ITERS} \
  -x PCIE_ORDER_MSG_MIN=${MSG_MIN} \
  -x PCIE_ORDER_MSG_MAX=${MSG_MAX} \
  -x PCIE_ORDER_GPU_VERIFY=${GPU_VERIFY} \
  -x LD_LIBRARY_PATH \
  -x PATH \
  ./rccl-UnitTestsMPI --gtest_filter="${TEST_FILTER}" 2>&1

echo "Exit code: $?"
