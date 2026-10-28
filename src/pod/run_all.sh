#!/bin/bash
# Runs on the rented GPU from /workspace/p04 (code + data uploaded by src/pod/launch.py). Order:
#   packages; in the background, the NYU part of the dataset and the model downloads (in run order);
#   the token-layout audit on the real processors; then each model in turn (extract.py), its weights deleted after.
# Progress: logs/*.log; out/<model>/ when a model is done; ALL_DONE at the end.
cd /workspace/p04
mkdir -p logs out models src_data/nyu
exec >> logs/run_all.log 2>&1
echo "start $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
pip install -q -r env/requirements-pod.txt && pip install -q --no-deps flash-linear-attention==0.5.2 fla-core==0.5.2
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
(
  curl -sL --retry 5 -o src_data/nyu/nyu_depth_v2_labeled.mat http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat &&
  python src/build_dataset.py nyu --src src_data/nyu --out data && python src/build_dataset.py finish --out data && touch data/READY
) > logs/data.log 2>&1 &
(
  grep -v '^#' src/pod/models.tsv | while IFS=$'\t' read -r name repo rev flags; do
    for try in 1 2 3; do
      hf download "$repo" --revision "$rev" --local-dir "models/$name" --exclude "*.gguf" > "logs/dl_$name.log" 2>&1 && touch "models/$name.ok" && break
    done
  done
) &
while [ ! -f data/READY ]; do sleep 10; if ! pgrep -f build_dataset.py > /dev/null && ! pgrep -f "curl -sL" > /dev/null && [ ! -f data/READY ]; then echo "dataset build failed"; tail -5 logs/data.log; sleep 5; [ -f data/READY ] || exit 1; fi; done
echo "data ready $(date -u +%FT%TZ)"; tail -3 logs/data.log
grep -v '^#' src/pod/models.tsv | while IFS=$'\t' read -r name repo rev flags; do
  while [ ! -f "models/$name.ok" ]; do sleep 10; done
  python src/layout_audit.py --model-dir "models/$name" --out "out/E00_audit_$name.json" > "logs/audit_$name.log" 2>&1
  echo "extract $name $(date -u +%FT%TZ)"
  python src/extract.py --model-dir "models/$name" --name "$name" --data data --out out $flags > "logs/extract_$name.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "extract $name failed; retrying without flash-linear-attention"; tail -20 "logs/extract_$name.log"
    pip uninstall -y -q flash-linear-attention fla-core
    python src/extract.py --model-dir "models/$name" --name "$name" --data data --out out $flags >> "logs/extract_$name.log" 2>&1 || { echo "extract $name failed again"; tail -20 "logs/extract_$name.log"; touch "out/$name.FAILED"; continue; }
  fi
  touch "out/$name.DONE"; rm -rf "models/$name"; echo "done $name $(date -u +%FT%TZ)"
done
touch ALL_DONE; echo "all done $(date -u +%FT%TZ)"
