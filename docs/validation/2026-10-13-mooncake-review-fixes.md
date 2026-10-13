# Mooncake review fixes — 2026-10-13

Two review findings were reproduced and fixed on the existing PR branch.

## Target teardown

Previously the target reference was cleared before close and no final exit status was checked. A simulated SIGSEGV during close allowed both ranks to return 0. The new regression fails on that original implementation, and both ranks fail on the fixed implementation.

NativeProcess now records terminate/kill actions actually sent to its root. Only Popen reaps the root's status; psutil.wait_procs is restricted to descendants, avoiding loss of the original root exit code. Mooncake checks the target before and after shutdown, rejects unexpected exits including -11/-6, and accepts graceful exit 0 or the corresponding owned termination status only after requesting shutdown. All ranks wait for successful target_done records before writing completed summaries. A target_exit.json preserves the evidence.

The POSIX integration test starts a CPU substitute target whose SIGTERM handler raises real SIGSEGV, with core dumps disabled. It confirms root exit -11, failure on both ranks, no completed summaries and cleanup. Additional tests cover simulated crash, early exit, missing ownership, normal termination and preservation of an independently exited root's code 7.

## Buffer size

Mooncake v0.3.13 calls check_total_buffer_size before choosing either target or initiator. It expands buffer_size to block_size × batch_size × threads when undersized. See [native source](https://github.com/kvcache-ai/Mooncake/blob/v0.3.13/mooncake-transfer-engine/example/transfer_engine_bench.cpp#L832-L842).

The pack now rejects undersized configurations before preflight/native startup. Tests cover the reported 64 KiB input, one byte below the 96 MiB default minimum, exact equality, one byte above and proof that neither preflight nor NativeProcess is called on rejection. The first hardware run's 256 MiB and 1 GiB inputs already exceeded the 96 MiB minimum; its recorded buffer flags are unchanged by this rule.

## Verification

- Windows / Python 3.13: 111 tests collected, 110 passed, one real POSIX signal test skipped.
- Zhengzhou Linux image / Python 3.10: all 111 tests passed in a separate --init container, including real SIGSEGV teardown and the existing DeepEP/SGLang lifecycle tests.
- A standalone native Mooncake target was started on the authorised BW150 node, with physical GPU 3, logical GPU 0, ACTIVE mlx5_0, a 256 MiB buffer and the same 0.3.13 distribution/image recorded on 2026-10-12. RDMA setup and VRAM allocation succeeded. Our terminate action produced root exit -15; the new stop_target check accepted it and archived target_exit.json. This is a native teardown check, not a throughput measurement.
- The second node's SSH connection timed out during this revision, so the full two-node throughput run was not repeated. The 2026-10-12 throughput record remains the earlier hardware evidence.
- Only containers created for this review run were stopped; logs/source remain in a separate review task directory. Previous validation artifacts were preserved.
