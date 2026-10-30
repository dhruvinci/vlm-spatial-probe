#!/bin/bash
# E04 on the rented GPU (experiments/E04-more-measures/PREREG.md), from /workspace/p04. Uploaded: src, env, data_e04 (the
# parts built locally and their images: kitti_obj, h4d, h4d_motion, coco, kitti_trk, objectron, and the Hypersim
# selection), used/{c01,c03}/data/dataset.json (E01 / E03 ids to exclude), src_data/hypersim/*.csv.
# Built here, because they need h5py: the NYU and Hypersim surface parts; then the merge into data_e04/dataset.json.
# Then, per model in order (a budget cut removes the last ones): feat, noimg, mis, ask (src/extract_e04.py).
cd /workspace/p04
mkdir -p logs out models src_data/nyu
exec >> logs/run_all.log 2>&1
echo "start $(date -u +%FT%TZ)"
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q -r env/requirements-pod.txt
(
  curl -sL --retry 5 -o src_data/nyu/nyu_depth_v2_labeled.mat http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat &&
  python src/build_e04.py nyu --src src_data --out data_e04 --cache used/c01 --cache-e03 used/c03 &&
  (cd data_e04 && python -I code/hypersim_truth.py) &&
  python src/build_e04.py finish --out data_e04 && touch data_e04/READY
) > logs/data.log 2>&1 &
(
  for spec in "Qwen3.8-27B Qwen/Qwen3.8-27B 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" \
              "SpatialAxiom-9B SpatialAxiom/SpatialAxiom-9B bfb2cc3508976c8a77f753a3c7ef0ca4e44e27d4" \
              "Qwen3.5-9B Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a" \
              "SpaceQwen2.5-VL-3B remyxai/SpaceQwen2.5-VL-3B-Instruct 332ff3fbc40d0e62af64b1bd2e0a606b3cca028e" \
              "Qwen2.5-VL-3B Qwen/Qwen2.5-VL-3B-Instruct 66285546d2b821cf421d4f5eb2576359d3770cd3"; do
    set -- $spec
    for try in 1 2 3; do hf download "$2" --revision "$3" --local-dir "models/$1" --exclude "*.gguf" > "logs/dl_$1.log" 2>&1 && touch "models/$1.ok" && break; done
  done
) &
run() {  # name model flags...
  local name=$1 model=$2; shift 2
  echo "extract $name $(date -u +%FT%TZ)"
  python src/extract_e04.py --model-dir "models/$model" --name "$name" --out out "$@" > "logs/extract_$name.log" 2>&1 \
    && { touch "out/$name.DONE"; echo "done $name $(date -u +%FT%TZ)"; } || { echo "extract $name failed"; tail -20 "logs/extract_$name.log"; touch "out/$name.FAILED"; }
}
waitdata() { while [ ! -f data_e04/READY ]; do sleep 10; if ! pgrep -f build_e04.py > /dev/null && ! pgrep -f hypersim_truth.py > /dev/null && ! pgrep -f "curl -sL" > /dev/null && [ ! -f data_e04/READY ]; then sleep 5; [ -f data_e04/READY ] || { echo "E04 data build failed"; tail -20 logs/data.log; return 1; }; fi; done; }
for m in Qwen3.8-27B SpatialAxiom-9B Qwen3.5-9B SpaceQwen2.5-VL-3B Qwen2.5-VL-3B; do
  while [ ! -f "models/$m.ok" ]; do sleep 10; done
  case $m in Qwen3.8-27B) gb=32;; *9B) gb=48;; *) gb=64;; esac
  waitdata && run "e04_$m" "$m" --data data_e04 --parts feat,noimg,mis,ask --batch 16 --gen-batch $gb
  rm -rf "models/$m"
done
touch ALL_DONE; echo "all done $(date -u +%FT%TZ)"
