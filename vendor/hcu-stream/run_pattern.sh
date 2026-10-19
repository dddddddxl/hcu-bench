#!/usr/bin/env bash
#===============================================================================
# DCU / HCU bandwidth & SMU 辅助测试统一入口（CI_test.sh 保持独立，不在此脚本内）
#
# 用法概览:
#   ./run_pattern.sh help
#   ./run_pattern.sh pattern <1-8> [--numa]
#   ./run_pattern.sh peak <bmz|nmz|zd1|zd2>
#   ./run_pattern.sh suite <lds>
#   ./run_pattern.sh smu <df-sweep|gc-sweep> [profile]
#   ./run_pattern.sh list
# 兼容旧接口:
#   ./run_pattern.sh -p <1-8> [-n|--numa]
#
# 调试:
#   --print-cmd | -P   执行前打印/跟踪实际命令（peak 打印整行；pattern/suite 对子脚本 bash -x）
#===============================================================================

VERSION="2.2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd)"
export ROOT

# --- 终端展示 -----------------------------------------------------------------
if [[ -t 1 ]]; then
  _C_HDR=$'\033[1;34m'
  _C_SUB=$'\033[0;36m'
  _C_OK=$'\033[0;32m'
  _C_WRN=$'\033[0;33m'
  _C_RST=$'\033[0m'
else
  _C_HDR= _C_SUB= _C_OK= _C_WRN= _C_RST=
fi

_ts() { date '+%H:%M:%S'; }
log_info()  { printf '%s%s%s %s\n'  "${_C_SUB}" "[info]" "${_C_RST}" "$*"; }
log_ok()    { printf '%s%s%s %s\n'  "${_C_OK}" "[ ok ]" "${_C_RST}" "$*"; }
log_warn()  { printf '%s%s%s %s\n'  "${_C_WRN}" "[warn]" "${_C_RST}" "$*"; }
log_err()   { printf '%s%s%s %s\n'  "${_C_WRN}" "[err ]" "${_C_RST}" "$*" >&2; }

section() {
  printf '\n%s── %s ──%s\n' "${_C_HDR}" "$*" "${_C_RST}"
}

# --- 打印实际命令（--print-cmd / -P）-------------------------------------------
# 任意位置传入 -P/--print-cmd 均可；在脚本末尾从 "$@" 中剥离后再进入 main。
PRINT_CMD=${RUN_PATTERN_PRINT_CMD:-0}

print_cmd_line() {
  [[ "$PRINT_CMD" -eq 1 ]] || return 0
  printf '%s[print-cmd]%s %s\n' "${_C_SUB}" "${_C_RST}" "$*" >&2
}

# 调用 hcu-stream / run，并在开启 flag 时打印完整命令行
hcu_run() {
  print_cmd_line "$(printf '%q ' "$HCUBENCH_BIN" "$@")"
  "$HCUBENCH_BIN" "$@"
}

# --- 二进制解析 ---------------------------------------------------------------
resolve_hcu_bin() {
  if [[ -n "${HCUBENCH_BIN:-}" ]]; then
    echo "$HCUBENCH_BIN"
    return
  fi
  local c
  for c in "$ROOT/hcu-stream" "$ROOT/run"; do
    if [[ -x "$c" ]]; then
      echo "$c"
      return
    fi
  done
  for c in "$ROOT/hcu-stream" "$ROOT/run"; do
    if [[ -f "$c" ]]; then
      echo "$c"
      return
    fi
  done
  echo "$ROOT/hcu-stream"
}

HCUBENCH_BIN="$(resolve_hcu_bin)"
export HCUBENCH_BIN

# --- 执行 test_script 内脚本（将 ./run 替换为实际二进制路径）---------------
run_wrapped_script() {
  local rel=$1
  local path="$ROOT/test_script/$rel"
  if [[ ! -f "$path" ]]; then
    log_err "找不到脚本: $path"
    return 1
  fi
  local bin_q
  bin_q=$(printf '%q' "$HCUBENCH_BIN")
  section "执行 $rel  (HCUBENCH_BIN=$bin_q)"
  if [[ "$PRINT_CMD" -eq 1 ]]; then
    print_cmd_line "sed 's#./run#<HCUBENCH_BIN>#' \"$(printf '%q' "$path")\" | (cd $(printf '%q' "$ROOT") && bash -x /dev/stdin)"
  fi
  if [[ "$PRINT_CMD" -eq 1 ]]; then
    if ! sed "s#\\.\\/run#${bin_q}#g" "$path" | (cd "$ROOT" && PS4='+ ' bash -x /dev/stdin); then
      log_err "脚本执行失败: $rel"
      return 1
    fi
  else
    if ! sed "s#\\.\\/run#${bin_q}#g" "$path" | (cd "$ROOT" && bash); then
      log_err "脚本执行失败: $rel"
      return 1
    fi
  fi
  log_ok "完成 $rel"
}

# --- Peak 套件（原 BMZ_peak.sh / NMZ_peak.sh）---------------------------------
cmd_peak_bmz() {
  section "Peak 套件: BMZ（原 BMZ_peak.sh）"
  export HSA_DISABLE_CACHE=0
  print_cmd_line "export HSA_DISABLE_CACHE=0"
  hcu_run bandwidth memory_readx4_x1 --clean
  hcu_run bandwidth memory_readx4_x4 --lds 32 --clean

  export HSA_DISABLE_CACHE=1
  print_cmd_line "export HSA_DISABLE_CACHE=1"
  hcu_run bandwidth memory_write --clean
  export HSA_DISABLE_CACHE=0
  print_cmd_line "export HSA_DISABLE_CACHE=0"
  hcu_run bandwidth memory_write --clean
  export HSA_DISABLE_CACHE=1
  print_cmd_line "export HSA_DISABLE_CACHE=1"
  hcu_run bandwidth memory_writex4_x4 --lds 32 --clean
  export HSA_DISABLE_CACHE=0
  print_cmd_line "export HSA_DISABLE_CACHE=0"
  hcu_run bandwidth memory_writex4_x4 --lds 32 --clean
  hcu_run bandwidth memory_copy --clean

  hcu_run bandwidth memory_to_ldsx4 --lds 4 --clean
  hcu_run bandwidth memory_to_ldsx4_x4 --lds 32 --clean
  hcu_run bandwidth lds_to_memory --lds 4 --clean
  hcu_run bandwidth matrix_read --clean
  hcu_run bandwidth matrix_read_x4 --lds 32 --clean
  hcu_run bandwidth matrix_write --clean
  hcu_run bandwidth matrix_copy --clean
  hcu_run bandwidth mapping_matrix_write --clean
  hcu_run bandwidth tiling_read --clean
  log_ok "peak bmz 完成"
}

cmd_peak_nmz() {
  section "Peak 套件: NMZ（原 NMZ_peak.sh）"
  hcu_run bandwidth memory_readx4_x4 --clean
  hcu_run bandwidth memory_writex4_x4 --clean
  hcu_run bandwidth memory_copy -w 16 --clean

  hcu_run bandwidth memory_to_ldsx4_x4 --lds 4 --clean
  hcu_run bandwidth lds_to_memory --lds 4 --clean

  hcu_run bandwidth matrix_readx4_x4 --clean
  hcu_run bandwidth matrix_writex4_x4 --clean
  hcu_run bandwidth matrix_copy --clean

  hcu_run bandwidth mapping_matrix_writex4_x4 --clean
  hcu_run bandwidth tiling_read_2w --clean
  log_ok "peak nmz 完成"
}

cmd_peak_zd1() {
    section "ZD/YY table1"
    export MODEL=1400
    print_cmd_line "export MODEL=1400"

    hcu_run bandwidth l1_read --clean
    hcu_run bandwidth l2_read --clean
    hcu_run bandwidth l2_write --clean
    hcu_run bandwidth lds_read --clean
    hcu_run bandwidth lds_write --clean

    hcu_run bandwidth memory_readx4_x1 --clean
    hcu_run bandwidth memory_write --clean
    hcu_run bandwidth memory_copy --clean

    hcu_run bandwidth memory_to_ldsx1 --lds 4 --clean
    hcu_run bandwidth lds_to_memory --lds 4 --clean


    hcu_run bandwidth matrix_read --clean
    hcu_run bandwidth matrix_write --clean
    hcu_run bandwidth matrix_copy --clean
    hcu_run bandwidth mapping_matrix_write --clean
    hcu_run bandwidth tiling_read --clean

  log_ok "peak zd/yy table1 完成"
}

cmd_peak_zd2() {
    section "ZD/YY table2"
    hcu_run bandwidth memory_readx4_x1 -w 8 --clean
    hcu_run bandwidth memory_write -w 8 --clean
    hcu_run bandwidth memory_copy -w 8 --clean

    hcu_run bandwidth matrix_readx4_x4 --clean
    hcu_run bandwidth matrix_writex4_x4 --clean
    hcu_run bandwidth matrix_copy --clean

    hcu_run bandwidth memory_readx4_x4 --clean

    hcu_run bandwidth mapping_matrix_writex4_x4 --clean
    hcu_run bandwidth tiling_read_2w --clean

  log_ok "peak zd/yy table2 完成"
}


cmd_peak() {
  local which=${1:-}
  case "$which" in
    bmz) cmd_peak_bmz ;;
    nmz) cmd_peak_nmz ;;
    zd1) cmd_peak_zd1 ;;
    zd2) cmd_peak_zd2 ;;
    *)
      log_err "peak 子命令应为: bmz | nmz | zd1 | zd2"
      return 2
      ;;
  esac
}

# --- Pattern 1–8 --------------------------------------------------------------
declare -A PATTERN_SUMMARY=(
  [1]="L1/L2/LDS、global/matrix、tiling；含 HSA cache 开关与 16wave"
  [2]="读/写/copy 与矩阵，多 wave（8/16）"
  [3]="读/写/copy 与矩阵，wave 8/16（精简矩阵块）"
  [4]="memory_read dwordx4/x2/x1 与多 wave 组合"
  [5]="读/写/copy 与 matrix 多 wave 扫描"
  [6]="read/write dwordx1/x2/x4 宽度与 x4 向量"
  [7]="多 LDS 占用（6–64）下 memory/matrix/lds 等"
  [8]="与 pattern3 类似：读/写/copy + matrix，wave 8/16"
)

cmd_pattern() {
  local id="${1:-}"
  shift || true
  local use_numa=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --numa|-n) use_numa=1 ;;
      --print-cmd|-P) ;; # 已在全局剥离；若仍传入则忽略
      *) log_warn "忽略未知参数: $1" ;;
    esac
    shift
  done

  if [[ ! "$id" =~ ^[1-8]$ ]]; then
    log_err "pattern 编号须为 1–8，收到: ${id:-<空>}"
    return 2
  fi

  local script="pattern${id}.sh"
  [[ "$use_numa" -eq 1 ]] && script="pattern${id}_numa.sh"
  log_info "Pattern ${id} (${PATTERN_SUMMARY[$id]:-})  numa=${use_numa}  → ${script}"
  run_wrapped_script "$script"
}

cmd_list() {
  printf '%srun_pattern.sh v%s — 用例一览%s\n' "${_C_HDR}" "$VERSION" "${_C_RST}"
  printf '%-10s %s\n' "命令" "说明"
  printf '%s\n' "--------------------------------------------------------------------------------"
  printf '%-10s %s\n' "pattern 1" "${PATTERN_SUMMARY[1]}"
  printf '%-10s %s\n' "pattern 2" "${PATTERN_SUMMARY[2]}"
  printf '%-10s %s\n' "pattern 3" "${PATTERN_SUMMARY[3]}"
  printf '%-10s %s\n' "pattern 4" "${PATTERN_SUMMARY[4]}"
  printf '%-10s %s\n' "pattern 5" "${PATTERN_SUMMARY[5]}"
  printf '%-10s %s\n' "pattern 6" "${PATTERN_SUMMARY[6]}"
  printf '%-10s %s\n' "pattern 7" "${PATTERN_SUMMARY[7]}"
  printf '%-10s %s\n' "pattern 8" "${PATTERN_SUMMARY[8]}"
  printf '%s\n' "--------------------------------------------------------------------------------"
  printf '%-10s %s\n' "peak bmz" "BMZ 峰值带宽序列（read/write/copy/LDS/matrix…）"
  printf '%-10s %s\n' "peak nmz" "NMZ 峰值带宽序列（x4 读写、matrix x4 等）"
  printf '%-10s %s\n' "peak zd1" "ZD table1：L1/L2/LDS、memory、matrix、tiling（MODEL=1400）"
  printf '%-10s %s\n' "peak zd2" "ZD table2：多 wave / matrix x4 / tiling_2w"
  printf '%-10s %s\n' "suite lds" "按 LDS 规模扫描 memory/matrix（原 lds.sh）"
  printf '%-10s %s\n' "smu …" "板级频率/DF/GC 扫频（需本机 SMU 工具与 sudo）"
  printf '%s\n' "--------------------------------------------------------------------------------"
  printf '环境变量 HCUBENCH_BIN=%s\n' "$HCUBENCH_BIN"
}

# --- SMU 扫频（原 test_script/DF_fix_*.sh / GC_fix.sh）-------------------------
# 环境变量:
#   SMU_TOOL_ROOT   默认 /data/tools/smu_bmz
#   HIPBENCH_ROOT   默认本仓库根目录；每轮测试前会 cd 到此目录
#   SMU_BENCH_CMD   每轮扫频后执行的命令（必填才会真实跑板）；未设置则 dry-run
# 日志:
#   SMU_LOG_FILE     默认 $HIPBENCH_ROOT/smu_sweep.log

_smu_run_monitor() {
  local smu=$1
  print_cmd_line "(cd $(printf '%q' "$smu") && sudo ./hcu_monitor --board_type 1)"
  (cd "$smu" && sudo ./hcu_monitor --board_type 1)
}

_smu_freq() {
  local smu=$1 dev=$2 ch=$3 mode=$4 mhz=$5
  print_cmd_line "(cd $(printf '%q' "$smu") && sudo ./hcu_frequency -d $dev -c $ch -m $mode -f $mhz)"
  (cd "$smu" && sudo ./hcu_frequency -d "$dev" -c "$ch" -m "$mode" -f "$mhz")
}

cmd_smu_df_sweep() {
  local profile=${1:-1.0}
  local smu="${SMU_TOOL_ROOT:-/data/tools/smu_bmz}"
  local bench="${HIPBENCH_ROOT:-$ROOT}"
  local logf="${SMU_LOG_FILE:-$bench/smu_df_sweep.log}"
  local df_init=1000
  case "$profile" in
    1.0|1_0) df_init=1000 ;;
    1.5|1_5) df_init=1500 ;;
    1.8|1_8) df_init=1800 ;;
    *)
      log_err "df-sweep profile 应为 1.0 | 1.5 | 1.8"
      return 2
      ;;
  esac

  section "SMU DF 初值 + GC 扫频 (原 DF_fix_${profile}, DF@ch2=${df_init}MHz) → $logf"
  if [[ ! -d "$smu" ]]; then
    log_err "SMU 目录不存在: $smu （可设置 SMU_TOOL_ROOT）"
    return 1
  fi
  if [[ -z "${SMU_BENCH_CMD:-}" ]]; then
    log_warn "未设置 SMU_BENCH_CMD，仅执行频率/监控步骤（dry-run 基准）。"
    log_warn "例: export SMU_BENCH_CMD='./run_pattern.sh peak bmz'"
  fi

  : >"$logf"
  # 与 DF_fix_*.sh 一致: 先写死 DF(通道2)，再在循环里扫 GC(通道0)
  _smu_freq "$smu" 0 2 2 "$df_init"
  _smu_run_monitor "$smu"

  local gc
  for ((gc = 1000; gc <= 1800; gc += 50)); do
    log_info "设置 GC=${gc} MHz (通道 0)" | tee -a "$logf"
    _smu_freq "$smu" 0 0 2 "$gc"
    _smu_run_monitor "$smu"
    (
      cd "$bench" || exit 1
      if [[ -n "${SMU_BENCH_CMD:-}" ]]; then
        log_info "执行 SMU_BENCH_CMD …" | tee -a "$logf"
        print_cmd_line "eval $(printf '%q' "$SMU_BENCH_CMD")  # SMU_BENCH_CMD"
        eval "$SMU_BENCH_CMD" >>"$logf" 2>&1
      else
        log_info "(dry-run) 未执行 SMU_BENCH_CMD" | tee -a "$logf"
      fi
    )
    echo "----------------------------------" >>"$logf"
  done
  log_ok "DF/GC 扫频流程结束，日志: $logf"
}

cmd_smu_gc_sweep() {
  local smu="${SMU_TOOL_ROOT:-/data/tools/smu_bmz}"
  local bench="${HIPBENCH_ROOT:-$ROOT}"
  local logf="${SMU_LOG_FILE:-$bench/smu_gc_sweep.log}"

  section "SMU GC 初值 + DF(通道2) 扫频（原 GC_fix.sh）→ $logf"
  if [[ ! -d "$smu" ]]; then
    log_err "SMU 目录不存在: $smu"
    return 1
  fi
  if [[ -z "${SMU_BENCH_CMD:-}" ]]; then
    log_warn "未设置 SMU_BENCH_CMD 时为 dry-run。"
  fi

  : >"$logf"
  _smu_freq "$smu" 0 0 2 1500
  _smu_run_monitor "$smu"

  local gc
  for ((gc = 1000; gc <= 1800; gc += 50)); do
    log_info "设置 DF(通道2)=${gc} MHz" | tee -a "$logf"
    _smu_freq "$smu" 0 2 2 "$gc"
    _smu_run_monitor "$smu"
    (
      cd "$bench" || exit 1
      if [[ -n "${SMU_BENCH_CMD:-}" ]]; then
        print_cmd_line "eval $(printf '%q' "$SMU_BENCH_CMD")  # SMU_BENCH_CMD"
        eval "$SMU_BENCH_CMD" >>"$logf" 2>&1
      fi
    )
    echo "----------------------------------" >>"$logf"
  done
  log_ok "GC/DF 扫频流程结束，日志: $logf"
}

cmd_smu() {
  local sub=${1:-}
  shift || true
  case "$sub" in
    df-sweep) cmd_smu_df_sweep "${1:-1.0}" ;;
    gc-sweep) cmd_smu_gc_sweep ;;
    *)
      log_err "smu 子命令: df-sweep [1.0|1.5|1.8] | gc-sweep"
      return 2
      ;;
  esac
}

cmd_suite() {
  local name=${1:-}
  case "$name" in
    lds) run_wrapped_script "lds.sh" ;;
    *)
      log_err "suite 子命令当前支持: lds"
      return 2
      ;;
  esac
}

# --- 帮助 ---------------------------------------------------------------------
usage() {
  cat <<EOF
${_C_HDR}run_pattern.sh${_C_RST} v${VERSION} — 统一测试调度（仓库: $ROOT）

${_C_SUB}新命令格式${_C_RST}
  ./run_pattern.sh help | --help | -h
  ./run_pattern.sh list
  ./run_pattern.sh pattern <1-8> [--numa|-n]
  ./run_pattern.sh peak <bmz|nmz|zd1|zd2>
  ./run_pattern.sh suite lds
  ./run_pattern.sh smu df-sweep [1.0|1.5|1.8]
  ./run_pattern.sh smu gc-sweep

${_C_SUB}打印实际执行的命令${_C_RST}
  在任意位置加上  -P  或  --print-cmd ：
  - peak：每条 hcu 调用前打印完整命令行
  - pattern / suite：对生成的子 shell 使用 bash -x（显示替换后的命令）
  - smu：打印 sudo hcu_frequency / hcu_monitor 及 eval SMU_BENCH_CMD
  也可用环境变量：export RUN_PATTERN_PRINT_CMD=1

${_C_SUB}兼容旧格式${_C_RST}
  ./run_pattern.sh -p <1-8> [-n|--numa]
  ./run_pattern.sh --print-cmd -p 3 -n

${_C_SUB}环境变量${_C_RST}
  HCUBENCH_BIN   基准二进制（默认: $ROOT/hcu-stream 或 ./run）
  SMU_TOOL_ROOT  SMU 工具目录（默认 /data/tools/smu_bmz）
  HIPBENCH_ROOT  扫频时每轮 cd 的目录（默认本仓库根）
  SMU_BENCH_CMD  smu df-sweep/gc-sweep 每轮执行的命令（未设置则 dry-run）
  SMU_LOG_FILE   扫频日志路径

${_C_SUB}说明${_C_RST}
  - pattern 1–8 对应 test_script/pattern{N}.sh；加 --numa 时使用 pattern{N}_numa.sh
  - CI 相关请仍使用 ${ROOT}/CI_test.sh（本脚本不包含）
EOF
}

# --- 旧版 -p 解析 -------------------------------------------------------------
legacy_main() {
  local X_VALUE="" use_numa=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --print-cmd|-P)
        shift
        ;;
      -p)
        shift
        X_VALUE="${1:-}"
        [[ -n "${1:-}" ]] && shift
        ;;
      -n|--numa)
        use_numa=1
        shift
        ;;
      *)
        log_warn "未识别参数（旧模式）: $1"
        shift
        ;;
    esac
  done

  if [[ -z "$X_VALUE" ]]; then
    log_err "旧模式需要: -p <1-8>"
    usage
    exit 2
  fi

  local args=("$X_VALUE")
  [[ "$use_numa" -eq 1 ]] && args+=(--numa)
  cmd_pattern "${args[@]}"
}

# --- 主入口 -------------------------------------------------------------------
main() {
  if [[ $# -eq 0 ]]; then
    usage
    exit 0
  fi

  if [[ "$PRINT_CMD" -eq 1 ]]; then
    log_info "已启用 --print-cmd：将打印/跟踪实际执行的命令"
  fi

  case "${1:-}" in
    -h|--help|help)
      usage
      exit 0
      ;;
    list)
      cmd_list
      exit 0
      ;;
    -p)
      shift
      legacy_main -p "$@"
      exit $?
      ;;
    pattern)
      shift
      cmd_pattern "$@"
      exit $?
      ;;
    peak)
      shift
      cmd_peak "$@"
      exit $?
      ;;
    suite)
      shift
      cmd_suite "$@"
      exit $?
      ;;
    smu)
      shift
      cmd_smu "$@"
      exit $?
      ;;
    *)
      # 若首参为数字 1-8，视作 pattern（便于简短调用）
      if [[ "$1" =~ ^[1-8]$ ]]; then
        cmd_pattern "$@"
        exit $?
      fi
      log_err "未知命令: $1"
      usage
      exit 2
      ;;
  esac
}

# 从参数列表中剥离 -P / --print-cmd（可出现在任意位置）
_RP_ARGS=()
for _rp_x in "$@"; do
  case "$_rp_x" in
    -P|--print-cmd)
      PRINT_CMD=1
      export PRINT_CMD
      export RUN_PATTERN_PRINT_CMD=1
      ;;
    *)
      _RP_ARGS+=("$_rp_x")
      ;;
  esac
done

main "${_RP_ARGS[@]}"
