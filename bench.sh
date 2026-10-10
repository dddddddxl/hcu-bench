#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${BENCH_PYTHON:-python3}
if [[ -z "${BENCH_PYTHON:-}" && -x "$HERE/.venv/bin/python" ]]; then
    PYTHON="$HERE/.venv/bin/python"
fi
if ! "$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10); import yaml' >/dev/null 2>&1; then
    echo 'Bench requires Python 3.10+ and PyYAML. Use a prepared container/control environment, or set BENCH_PYTHON.' >&2
    exit 2
fi
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

case "${1:-}" in
    list|check|plan|run|status|stop|report|--version|-h|--help)
        exec "$PYTHON" -m hcu_bench "$@"
        ;;
esac

CONFIG="$HERE/configs/bench.yaml"
if [[ "${1:-}" == *.yaml || "${1:-}" == *.yml ]]; then
    CONFIG=$1
    shift
fi
SUITE=${1:-all}
if (( $# )); then shift; fi
exec "$PYTHON" -m hcu_bench run -c "$CONFIG" --suite "$SUITE" "$@"
