We evaluated the performance of our sandbox API against prime_code local evaluation method. Our results demonstrate that it's more efficent and scalable, which is highly suitable for CodeRL training workloads.

## Experimental Setup

### Dataset
- **Source**: PrimeIntellect/verifiable-coding-problems
- **Sample Size**: 2,048 Python problem instances with reference code

### Hardware Environment
- **CPU**: Intel Xeon Platinum 8378A @ 3.00GHz
- **Cores**: 128 physical cores
- **Memory**: 1 TB

### Sandbox Configuration
- **Load Balancer**: NGINX
- **Unit Test Parallelism**: `max_runner_concurrency=16`

### Client Configuration
- **Request Concurrency**: 128
- **Batch Processing**: Enabled via `common_evaluate_batch`

## Performance Results

Our results demonstrate that the sandbox API achieves a **14.57% performance improvement** over the baseline, with the ability to scale linearly across multiple nodes. With two sandbox nodes, we achieved a **38.29% reduction** in evaluation time.

| Evaluation Method | Time (s) | Speed-up | Client Concurrency | Workers | Total CPU Cores |
|-------------------|----------|----------|-------------------|---------|-----------------|
| **Baseline** (No Sandbox) | 164.89 | - | 128 | - | 128 |
| **Sandbox** (1 node) | 140.86 | 14.57% | 128 | 32 | 128 |
| **Sandbox** (2 nodes) | 86.93 | 47.28% | 128 | 64 | 256 |
| **Sandbox** (3 nodes) | 66.84 | 58.06% | 128 | 96 | 384 |