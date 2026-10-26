#!/bin/bash
# Download the public benchmarks (sycophancy-eval, FaithEval, RAMDocs) into $CREDIT_ROOT/data/bench.
. "$(dirname "$0")/common.sh"
$PY credit.fetch_data
