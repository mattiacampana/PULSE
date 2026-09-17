# PULSE portable math tests

This directory tests the allocation-free, platform-neutral PULSE core without changing the
Zephyr build. From the firmware repository root, a C99 host compiler can run it with:

```sh
gcc -std=c99 -O2 -Wall -Wextra -Werror -pedantic -Isrc/pulse \
    src/pulse/pulse_math.c src/pulse/pulse_peers.c \
    tests/pulse_math/test_pulse_math.c -lm -o pulse_math_test
./pulse_math_test
```

The analytic fixture activates one path through both convolutions and the projection, then checks
the batch-16 mean cross-entropy gradient in canonical `[W1,b1,W2,b2]` order. Other checks cover
two-step local SGD, zero/positive/negative cosine agreement, Python-reference norm scaling,
agreement-calibrated mixing, signed utility EMA, UCB selection, conservative no-contact ties,
seeded tie-breaking, and fixed-table capacity.

For final embedded/reference validation, export real PyTorch tensors as FP32 arrays and compare
the full gradient and updated head. Floating-point accumulation order, the host math library, and
Arm compiler options can cause small last-bit differences; compare cosine similarity and maximum
absolute error rather than requiring byte identity.

When PyTorch, NumPy, and a host C compiler are available, the deterministic cross-check automates
that comparison against the exact Conv1d/MLP graph:

```sh
python tests/pulse_math/compare_pytorch.py
```
