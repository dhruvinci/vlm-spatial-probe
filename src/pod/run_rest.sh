#!/bin/bash
# Takes over from run_all.sh mid-run with a new model order (the SpaceQwen pair before SpatialAxiom, so a budget cut
# would fall on SpatialAxiom first). Waits for the running extraction (PID $1, model $2) to exit, marks it done if it
# finished, then runs the remaining models in the order given in src/pod/models_rest.tsv. With PID "-", it just runs
# the list.
cd /workspace/p04
exec >> logs/run_all.log 2>&1
echo "run_rest takes over $(date -u +%FT%TZ): waiting for PID $1 ($2)"
[ "$1" = "-" ] || while kill -0 "$1" 2>/dev/null; do sleep 10; done
if [ "$1" = "-" ]; then :
elif python -c "import json,sys; sys.exit(0 if 'total_s' in json.load(open('out/$2/meta.json')) else 1)"; then
  touch "out/$2.DONE"; rm -rf "models/$2"; echo "done $2 $(date -u +%FT%TZ)"
else
  echo "$2 did not finish"; touch "out/$2.FAILED"
fi
grep -v '^#' src/pod/models_rest.tsv | while IFS=$'\t' read -r name repo rev flags; do
  while [ ! -f "models/$name.ok" ]; do sleep 10; done
  python src/layout_audit.py --model-dir "models/$name" --out "out/E00_audit_$name.json" > "logs/audit_$name.log" 2>&1
  echo "extract $name $(date -u +%FT%TZ)"
  python src/extract.py --model-dir "models/$name" --name "$name" --data data --out out $flags > "logs/extract_$name.log" 2>&1 \
    || { echo "extract $name failed"; tail -20 "logs/extract_$name.log"; touch "out/$name.FAILED"; continue; }
  touch "out/$name.DONE"; rm -rf "models/$name"; echo "done $name $(date -u +%FT%TZ)"
done
touch ALL_DONE; echo "all done $(date -u +%FT%TZ)"
