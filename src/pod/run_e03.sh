#!/bin/bash
# E03 + E06 on the rented GPU, from /workspace/p04 (uploaded: src, env, data = E01's dataset and images,
# data_e03/h4d_more.json + its images, src_data/kitti/{labels,calib}.zip, src_data/hypersim/*.csv).
# E03 data (NYU superset, KITTI superset, Hypersim) is built here; then, per model in order:
#   Qwen3.8-27B and SpatialAxiom-9B: E06 (ask2, yesno2 on E01's images) and E03 (with-image and no-image passes on the
#   new images; spoken answers on Hypersim only); Qwen3.5-9B, SpaceQwen2.5-VL-3B, Qwen2.5-VL-3B: E06 only.
cd /workspace/p04
mkdir -p logs out models src_data/nyu
exec >> logs/run_all.log 2>&1
echo "start $(date -u +%FT%TZ)"
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q -r env/requirements-pod.txt
(
  curl -sL --retry 5 -o src_data/nyu/nyu_depth_v2_labeled.mat http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat &&
  python src/build_dataset.py nyu --src src_data/nyu --out data_e03 --n 2000 &&
  python src/build_dataset.py kitti --src src_data/kitti --out data_e03 --n 1000 &&
  python src/build_dataset.py hypersim --src src_data/hypersim --out data_e03 &&
  python src/build_dataset.py finish_e03 --out data_e03 --exclude data/dataset.json && touch data_e03/READY
) > logs/data_e03.log 2>&1 &
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
  python src/extract.py --model-dir "models/$model" --name "$name" --out out "$@" > "logs/extract_$name.log" 2>&1 \
    && { touch "out/$name.DONE"; echo "done $name $(date -u +%FT%TZ)"; } || { echo "extract $name failed"; tail -20 "logs/extract_$name.log"; touch "out/$name.FAILED"; }
}
waitdata() { while [ ! -f data_e03/READY ]; do sleep 10; if ! pgrep -f build_dataset.py > /dev/null && ! pgrep -f "curl -sL" > /dev/null && [ ! -f data_e03/READY ]; then sleep 5; [ -f data_e03/READY ] || { echo "E03 data build failed"; tail -20 logs/data_e03.log; return 1; }; fi; done; }
for m in Qwen3.8-27B SpatialAxiom-9B Qwen3.5-9B SpaceQwen2.5-VL-3B Qwen2.5-VL-3B; do
  while [ ! -f "models/$m.ok" ]; do sleep 10; done
  case $m in Qwen3.8-27B) gb=32;; *9B) gb=96;; *) gb=128;; esac
  run "e06_$m" "$m" --data data --parts ask2,yesno2 --batch 16 --gen-batch $gb
  if [ "$m" = "Qwen3.8-27B" ] || [ "$m" = "SpatialAxiom-9B" ]; then
    waitdata && run "e03_$m" "$m" --data data_e03 --parts feat,noimg,ask --ask-domains indoor_synth --batch 16 --gen-batch $gb
  fi
  rm -rf "models/$m"
done
touch ALL_DONE; echo "all done $(date -u +%FT%TZ)"
