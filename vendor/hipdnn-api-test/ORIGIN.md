# DTK hipDNN API test source

`dtk-api-test-source.tar.gz` contains the original `api_test/CMakeLists.txt`, README and complete `cpp` tree used for the DTK-aligned Conv reference. The archive is retained byte for byte; no build output or DTK libraries are included. Its digest is recorded in `vendor/sources.json`.

The original CMake configuration references multiple API sources, so the complete tree is retained. hcu-bench builds only the `test_conv` target and executes the original Conv command set.

The archive has no root LICENSE. Original per-file copyright and MIT SPDX notices are retained; a blanket license for the remaining files is not asserted. Repository bundling was requested by the user on 2026-11-05. This does not relicense the original files.
