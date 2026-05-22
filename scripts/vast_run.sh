#!/usr/bin/env bash
# Automate vast.ai workflow: search offers, launch instance, wait for SSH,
# rsync this repo, bootstrap env, run the smoke training, tear down.
#
# State file: .vast/instance.json (gitignored). Holds id + ssh details.
#
# Subcommands:
#   search                List candidate GPU offers.
#   launch [OFFER_ID]     Create + start instance (defaults to cheapest A6000).
#   status                Show current instance state (actual_status, ssh, $/hr).
#   wait                  Poll until actual_status=running and SSH responds.
#   ssh [-- CMD...]       SSH interactive (or run CMD remotely).
#   rsync                 Push repo to /workspace/rsdiff on instance.
#   bootstrap             Run scripts/vast_setup.sh --download-rsicd remotely.
#   run [EPOCHS] [NAME]   Launch LR training (legacy/run_smoke.sh) via nohup; survives disconnect.
#                         Default: 10 epochs, LOG_NAME=smoke_lr_gdm. Use 1000 + full_lr_gdm for full run.
#   run-sr [EPOCHS] [NAME]  Launch SR training (legacy/run_sr.sh, path B) via nohup.
#   run-joint [EPOCHS] [NAME]  Launch joint fine-tune (legacy/run_joint.sh, both unets) via nohup.
#                         Default: 1000 epochs, LOG_NAME=full_sr_gdm. Seeds+freezes LR base.
#                         Env: LR_CKPT overrides the base checkpoint to seed unet 1.
#   logs [NAME]           Tail logfile.log from legacy/DDPM/logs/<NAME>/ (default smoke_lr_gdm).
#   gpu                   Show clean nvidia-smi snapshot (vram, util, power, temp).
#   pull [SUBPATH]        Rsync remote artifacts (ckpt + samples + logs) -> outputs/vast/.
#                         Default SUBPATH=legacy/DDPM/logs/smoke_lr_gdm.
#   snapshot [NAME]       One-shot: copy checkpoint.pt -> checkpoint_step{N}.pt.
#                         Defaults NAME=full_lr_gdm, MAX=20, PRUNE_TO=10.
#                         Also writes milestones/ckpt_step{N}.pt every 100 epochs.
#                         Env: SNAP_WATCH=SECONDS spawns nohup watcher loop.
#                              SNAP_MAX, SNAP_PRUNE_TO, MILESTONE_STEPS override defaults.
#   pull-milestones [NAME]  Rsync legacy/DDPM/logs/<NAME>/milestones/ -> outputs/vast/.../milestones/.
#   sample-grid [NAME] [STEP]  Sample 16 test captions from a ckpt on remote.
#                              Default: latest milestone if any, else checkpoint.pt.
#                              STEP picks milestones/ckpt_step{STEP}.pt explicitly.
#                              SR auto-detected for *sr* names (256 cascade); SR=1/0 to force.
#   destroy               Tear instance down (asks unless FORCE=1). Tip: pull first.
#   swap OFFER_ID         FORCE-destroy current + launch new offer + wait + rsync + bootstrap.
#   all [OFFER_ID]        launch -> wait -> rsync -> bootstrap -> run (10-epoch smoke).
#
# Requires: vastai (uv tool install vastai), rsync, ssh, jq optional.
# Loads VAST_API_KEY from ./.env automatically if present.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STATE_DIR="${REPO_ROOT}/.vast"
# VAST_STATE overrides the state file so a second instance (e.g. FID-gen box) can
# be managed in parallel without clobbering the training instance's state:
#   VAST_STATE=fid ./scripts/vast_run.sh launch ...  -> .vast/fid.json
STATE_FILE="${STATE_DIR}/${VAST_STATE:+${VAST_STATE}.}instance.json"
mkdir -p "${STATE_DIR}"

IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel}"
DISK_GB="${VAST_DISK_GB:-60}"
REMOTE_ROOT="${VAST_REMOTE_ROOT:-/workspace/rsdiff}"
GPU_FILTER="${VAST_GPU_FILTER:-gpu_name in [RTX_A6000,A40,RTX_4090,A100_SXM4,RTX_3090]}"
SEARCH_QUERY="${VAST_SEARCH_QUERY:-${GPU_FILTER} num_gpus=1 reliability>0.95 verified=true rentable=true cuda_vers>=12.1 inet_down>200 disk_space>=80}"

if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "${REPO_ROOT}/.env"
  set +a
  if [ -n "${VAST_API_KEY:-}" ]; then
    vastai set api-key "${VAST_API_KEY}" >/dev/null 2>&1 || true
  fi
fi

state_get() {
  python3 -c "import json,sys;d=json.load(open('${STATE_FILE}'));print(d.get('$1',''))" 2>/dev/null
}

require_state() {
  if [ ! -f "${STATE_FILE}" ]; then
    echo "no instance recorded. run: $0 launch" >&2; exit 1
  fi
}

ssh_cmd() {
  local host port
  host="$(state_get ssh_host)"
  port="$(state_get ssh_port)"
  if [ -z "${host}" ] || [ -z "${port}" ]; then
    echo "ssh details missing — run: $0 wait" >&2; exit 1
  fi
  ssh -o StrictHostKeyChecking=accept-new -p "${port}" "root@${host}" "$@"
}

cmd_search() {
  vastai search offers "${SEARCH_QUERY}" -o 'dph_total' --raw | python3 -c "
import sys,json
offers=json.loads(sys.stdin.read())
print(f'candidates: {len(offers)}')
for o in offers[:12]:
    print(f\"  id={o['id']:>9}  \${o['dph_total']:.3f}/hr  R={o['reliability2']:.3f}  cuda={o['cuda_max_good']}  vram={o['gpu_ram']/1024:.1f}GB  gpu={o['gpu_name']:<14}  net_down={o['inet_down']:.0f}  loc={o.get('geolocation','?')}\")
"
}

cmd_launch() {
  local offer_id="${1:-}"
  if [ -z "${offer_id}" ]; then
    offer_id=$(vastai search offers "${SEARCH_QUERY}" -o 'dph_total' --raw \
      | python3 -c "import sys,json;o=json.loads(sys.stdin.read());print(o[0]['id']) if o else exit(1)")
    [ -z "${offer_id}" ] && { echo "no offers" >&2; exit 1; }
    echo "picked cheapest: ${offer_id}"
  fi
  local out contract
  out=$(vastai create instance "${offer_id}" --image "${IMAGE}" --disk "${DISK_GB}" --ssh --direct 2>&1)
  echo "${out}"
  contract=$(echo "${out}" | python3 -c "
import sys,re,ast
m=re.search(r'\{.*\}', sys.stdin.read())
if not m: sys.exit('no contract id in output')
d=ast.literal_eval(m.group(0))
print(d.get('new_contract',''))" 2>/dev/null) || true
  if [ -z "${contract}" ]; then
    echo "could not parse contract id" >&2; exit 1
  fi
  python3 -c "import json;json.dump({'id':${contract}},open('${STATE_FILE}','w'))"
  echo "contract: ${contract}"
  vastai start instance "${contract}" 2>&1 || true
  # Attach local pubkey so SSH works without depending on the global key on file.
  local pub="${SSH_PUBKEY_FILE:-${HOME}/.ssh/id_ed25519.pub}"
  if [ -f "${pub}" ]; then
    vastai attach ssh "${contract}" "$(cat "${pub}")" >/dev/null 2>&1 \
      && echo "attached ${pub} to ${contract}" \
      || echo "warn: vastai attach ssh failed (key may already be present)"
  fi
  echo "state -> ${STATE_FILE}"
}

cmd_status() {
  require_state
  local id; id="$(state_get id)"
  vastai show instances --raw | python3 -c "
import sys,json
for i in json.loads(sys.stdin.read()):
    if i['id'] == ${id}:
        print(f\"id={i['id']} actual={i.get('actual_status')} cur_state={i.get('cur_state')} intended={i.get('intended_status')} ssh={i.get('ssh_host')}:{i.get('ssh_port')} \${i.get('dph_total',0):.3f}/hr\")
        break
"
}

cmd_wait() {
  require_state
  local id; id="$(state_get id)"
  echo "waiting for instance ${id} ssh ready (direct endpoint preferred)..."
  while true; do
    info=$(vastai show instance "${id}" --raw | python3 -c "
import sys,json
i=json.loads(sys.stdin.read())
print(f\"{i.get('actual_status')}|{i.get('public_ipaddr','')}|{i.get('direct_port_start','')}|{i.get('ssh_host','')}|{i.get('ssh_port','')}\")
")
    IFS='|' read -r actual ip dport phost pport <<< "${info}"
    echo "  actual=${actual} direct=${ip}:${dport} proxy=${phost}:${pport}"
    if [ "${actual}" = "running" ] && [ -n "${ip}" ] && [ -n "${dport}" ] && [ "${dport}" != "None" ]; then
      python3 -c "
import json
d=json.load(open('${STATE_FILE}'))
d.update({'ssh_host':'${ip}','ssh_port':${dport}})
json.dump(d, open('${STATE_FILE}','w'))
"
      sleep 5
      if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -p "${dport}" "root@${ip}" 'echo ok' 2>/dev/null | grep -q ok; then
        echo "ssh direct ok: ${ip}:${dport}"
        return 0
      else
        echo "  ssh not yet accepting; retrying..."
      fi
    fi
    sleep 15
  done
}

cmd_ssh() {
  require_state
  ssh_cmd "$@"
}

cmd_rsync() {
  require_state
  local host port
  host="$(state_get ssh_host)"
  port="$(state_get ssh_port)"
  rsync -avz -e "ssh -p ${port} -o StrictHostKeyChecking=accept-new" \
    --exclude '.git' --exclude '.venv*' --exclude '__pycache__' \
    --exclude 'data/' --exclude 'outputs/' --exclude '.vast/' \
    --exclude '*.safetensors' --exclude '*.pt' --exclude '*.ckpt' \
    --exclude '.env' --exclude '.env.local' --exclude 'secrets/' \
    --exclude '.claude/settings.local.json' --exclude '.claude/.memory/' \
    "${REPO_ROOT}/" "root@${host}:${REMOTE_ROOT}/"
}

cmd_bootstrap() {
  require_state
  ssh_cmd "cd ${REMOTE_ROOT} && bash scripts/vast_setup.sh --download-rsicd"
}

cmd_run() {
  require_state
  local epochs="${1:-10}"
  local name="${2:-smoke_lr_gdm}"
  local ts="$(date +%Y%m%d_%H%M%S)"
  ssh_cmd "mkdir -p ${REMOTE_ROOT}/outputs && cd ${REMOTE_ROOT} && \
    EPOCHS=${epochs} LOG_NAME=${name} nohup bash legacy/run_smoke.sh \
      > ${REMOTE_ROOT}/outputs/run_${name}_${ts}.log 2>&1 < /dev/null & \
    echo started run pid=\$! name=${name} epochs=${epochs} \
      stdout_log=outputs/run_${name}_${ts}.log \
      logger_log=legacy/DDPM/logs/${name}/logfile.log"
}

cmd_run_sr() {
  require_state
  local epochs="${1:-1000}"
  local name="${2:-full_sr_gdm}"
  local ts="$(date +%Y%m%d_%H%M%S)"
  ssh_cmd "mkdir -p ${REMOTE_ROOT}/outputs && cd ${REMOTE_ROOT} && \
    EPOCHS=${epochs} LOG_NAME=${name} ${LR_CKPT:+LR_CKPT='${LR_CKPT}'} nohup bash legacy/run_sr.sh \
      > ${REMOTE_ROOT}/outputs/run_${name}_${ts}.log 2>&1 < /dev/null & \
    echo started SR run pid=\$! name=${name} epochs=${epochs} \
      stdout_log=outputs/run_${name}_${ts}.log \
      logger_log=legacy/DDPM/logs/${name}/logfile.log"
}

cmd_run_joint() {
  require_state
  local epochs="${1:-200}"
  local name="${2:-full_joint_gdm}"
  local ts="$(date +%Y%m%d_%H%M%S)"
  ssh_cmd "mkdir -p ${REMOTE_ROOT}/outputs && cd ${REMOTE_ROOT} && \
    EPOCHS=${epochs} LOG_NAME=${name} ${INIT_CKPT:+INIT_CKPT='${INIT_CKPT}'} ${LAMBDA_SR:+LAMBDA_SR='${LAMBDA_SR}'} nohup bash legacy/run_joint.sh \
      > ${REMOTE_ROOT}/outputs/run_${name}_${ts}.log 2>&1 < /dev/null & \
    echo started joint run pid=\$! name=${name} epochs=${epochs} \
      stdout_log=outputs/run_${name}_${ts}.log \
      logger_log=legacy/DDPM/logs/${name}/logfile.log"
}

cmd_logs() {
  require_state
  local name="${1:-smoke_lr_gdm}"
  ssh_cmd "tail -n 60 ${REMOTE_ROOT}/legacy/DDPM/logs/${name}/logfile.log"
}

cmd_gpu() {
  require_state
  ssh_cmd "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem --format=csv,noheader,nounits" 2>/dev/null | \
  awk -F', *' '
  /^[A-Za-z]/ {
    name=$1; vmem_used=$2; vmem_tot=$3; gpu_u=$4; mem_u=$5; pwr=$6; pwr_max=$7; t=$8; clk_sm=$9; clk_mem=$10;
    vram_pct=(vmem_used/vmem_tot)*100;
    pwr_pct=(pwr/pwr_max)*100;
    printf "  GPU            %s\n", name;
    printf "  ----------------------------------------------\n";
    printf "  VRAM           %5d / %-5d MiB   (%5.1f%%)\n", vmem_used, vmem_tot, vram_pct;
    printf "  GPU util       %3d%%\n", gpu_u;
    printf "  Mem-BW util    %3d%%\n", mem_u;
    printf "  Power          %3d / %-3d W       (%5.1f%%)\n", pwr, pwr_max, pwr_pct;
    printf "  Temperature    %3d C\n", t;
    printf "  Clocks SM/Mem  %4d / %-4d MHz\n", clk_sm, clk_mem;
  }'
}

cmd_snapshot() {
  require_state
  local name="${1:-full_lr_gdm}"
  local max="${SNAP_MAX:-20}"
  local prune_to="${SNAP_PRUNE_TO:-10}"
  local watch="${SNAP_WATCH:-0}"
  local logdir="${REMOTE_ROOT}/legacy/DDPM/logs/${name}"
  if [ "${watch}" -eq 0 ]; then
    ssh_cmd "bash ${REMOTE_ROOT}/scripts/remote_snapshot.sh '${logdir}' '${max}' '${prune_to}'"
  else
    ssh_cmd "nohup bash -c 'while true; do bash ${REMOTE_ROOT}/scripts/remote_snapshot.sh \"${logdir}\" \"${max}\" \"${prune_to}\" 2>&1; sleep ${watch}; done' > ${REMOTE_ROOT}/outputs/snap_watcher_${name}.log 2>&1 < /dev/null & echo started watcher pid=\$! interval=${watch}s max=${max} prune_to=${prune_to} log=outputs/snap_watcher_${name}.log"
  fi
}

cmd_pull() {
  require_state
  local sub="${1:-legacy/DDPM/logs/smoke_lr_gdm}"
  local host port dest
  host="$(state_get ssh_host)"
  port="$(state_get ssh_port)"
  dest="${REPO_ROOT}/outputs/vast/${sub}"
  mkdir -p "${dest}"
  rsync -avz --partial -e "ssh -p ${port} -o StrictHostKeyChecking=accept-new" \
    "root@${host}:${REMOTE_ROOT}/${sub}/" "${dest}/"
  echo "pulled -> ${dest}"
}

cmd_sample_grid() {
  require_state
  local name="${1:-full_lr_gdm}"
  local step="${2:-}"
  local n="${SAMPLE_N:-16}"
  local cols="${SAMPLE_COLS:-4}"
  local scale="${COND_SCALE:-4.0}"
  local batch_arg=""
  [ -n "${SAMPLE_BATCH:-}" ] && batch_arg="--batch ${SAMPLE_BATCH}"
  local grid_arg=""
  [ "${NO_GRID:-0}" = "1" ] && grid_arg="--no_grid"
  local logdir="${REMOTE_ROOT}/legacy/DDPM/logs/${name}"
  local ckpt_arg=""
  local out_subdir=""
  if [ -n "${step}" ]; then
    ckpt_arg="--ckpt ${logdir}/milestones/ckpt_step${step}.pt"
    out_subdir="--out_subdir grid_step${step}"
  fi
  # Two-unet cascade for SR runs (full_sr_gdm / *_sr_*). Set SR=0 to force base-only.
  local sr_arg=""
  case "${SR:-auto}" in
    1) sr_arg="--sr" ;;
    0) sr_arg="" ;;
    *) [[ "${name}" == *sr* ]] && sr_arg="--sr" ;;
  esac
  ssh_cmd "cd ${REMOTE_ROOT} && python legacy/DDPM/sample_grid.py \
    --log_dir ${logdir} \
    --data_root ${REMOTE_ROOT}/data/RSICD_optimal \
    --n ${n} --cols ${cols} --cond_scale ${scale} \
    --split test --device auto \
    ${batch_arg} ${grid_arg} ${sr_arg} ${ckpt_arg} ${out_subdir}"
}

cmd_pull_milestones() {
  require_state
  local name="${1:-full_lr_gdm}"
  local host port dest remote_dir
  host="$(state_get ssh_host)"
  port="$(state_get ssh_port)"
  dest="${REPO_ROOT}/outputs/vast/legacy/DDPM/logs/${name}/milestones"
  remote_dir="${REMOTE_ROOT}/legacy/DDPM/logs/${name}/milestones"
  mkdir -p "${dest}"
  ssh_cmd "mkdir -p '${remote_dir}'"
  rsync -avz --partial -e "ssh -p ${port} -o StrictHostKeyChecking=accept-new" \
    "root@${host}:${remote_dir}/" "${dest}/"
  local n; n=$(find "${dest}" -maxdepth 1 -name '*.pt' -type f 2>/dev/null | wc -l | tr -d ' ')
  echo "milestones local: ${n} files -> ${dest}"
}

cmd_destroy() {
  require_state
  local id; id="$(state_get id)"
  if [ "${SKIP_PULL:-0}" != "1" ]; then
    echo "tip: run '$0 pull' first if you want artifacts. (SKIP_PULL=1 to silence)"
  fi
  if [ "${FORCE:-0}" = "1" ]; then
    yes | vastai destroy instance "${id}" 2>&1 || true
  else
    vastai destroy instance "${id}"
  fi
  rm -f "${STATE_FILE}"
}

cmd_all() {
  cmd_launch "$@"
  cmd_wait
  cmd_rsync
  cmd_bootstrap
  cmd_run
}

cmd_swap() {
  local offer_id="${1:-}"
  if [ -z "${offer_id}" ]; then
    echo "usage: $0 swap OFFER_ID" >&2; exit 1
  fi
  if [ -f "${STATE_FILE}" ]; then
    echo "destroying current instance (FORCE=1)..."
    FORCE=1 cmd_destroy || true
  fi
  cmd_launch "${offer_id}"
  cmd_wait
  cmd_rsync
  cmd_bootstrap
}

sub="${1:-help}"
shift || true
case "${sub}" in
  search)    cmd_search "$@" ;;
  launch)    cmd_launch "$@" ;;
  status)    cmd_status "$@" ;;
  wait)      cmd_wait "$@" ;;
  ssh)       cmd_ssh "$@" ;;
  rsync)     cmd_rsync "$@" ;;
  bootstrap) cmd_bootstrap "$@" ;;
  run)       cmd_run "$@" ;;
  run-sr)    cmd_run_sr "$@" ;;
  run-joint) cmd_run_joint "$@" ;;
  logs)      cmd_logs "$@" ;;
  gpu)       cmd_gpu "$@" ;;
  pull)      cmd_pull "$@" ;;
  pull-milestones) cmd_pull_milestones "$@" ;;
  sample-grid) cmd_sample_grid "$@" ;;
  snapshot)  cmd_snapshot "$@" ;;
  destroy)   cmd_destroy "$@" ;;
  swap)      cmd_swap "$@" ;;
  all)       cmd_all "$@" ;;
  help|*)
    awk '/^# Subcommands:/,/^$/' "$0" | sed -E 's/^#( |$)//'
    ;;
esac
