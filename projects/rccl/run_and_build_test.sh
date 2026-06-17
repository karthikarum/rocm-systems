#!/bin/bash
# PCIe Ordering Stress Test — build & run script
# Requires multi-node (>= 2 nodes) with AINIC NICs

RCCL_HOME=/apps/shared/karthik/repo/integration/rccl-rocm-sys/rocm-systems/projects/rccl

# build test command:
cd ${RCCL_HOME}
sudo MPI_PATH=/apps/shared/karthik/repo/ompi/install ./install.sh --debug --enable-mpi-tests -t --amdgpu_targets gfx942

# run test command:
# --- Node configuration ---
# Minimum 2 nodes; adjust NP to match the number of nodes in NODES.
# Minimum 4 nodes required (tests need kMinFourProcesses=4)
NODES=smc300x-ccs-aus-gpud334,smc300x-ccs-aus-gpuf2ba,smc300x-ccs-aus-gpuf2ae,smc300x-ccs-aus-gpufccf
NP=4

# --- MPI transport configuration ---
# - btl ^openib : disable OpenIB BTL for MPI control traffic (avoids ionic_0 warnings/stalls)
# - btl_tcp_if_include ens50f0 : use the management network (10.235.x.x) for MPI TCP control
# These settings only affect MPI's own communication; RDMA data transfer is handled
# directly by the rocmNetIb plugin via libibverbs.
MPI_TRANSPORT_FLAGS="\
  --mca btl ^openib \
  --mca btl_tcp_if_include ens50f0 \
  --mca btl_openib_warn_no_device_params_found 0"

cd ${RCCL_HOME}/build/debug/test

mpirun -np ${NP} -H ${NODES} -npernode 1 --bind-to numa \
  ${MPI_TRANSPORT_FLAGS} \
  -x NCCL_GDR_FLUSH_DISABLE=1 \
  ./rccl-UnitTestsMPI --gtest_filter='*PCIeOrdering*'
