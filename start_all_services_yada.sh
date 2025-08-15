#!/bin/bash

# Absolute path to where you want to cd
WORKDIR="OpenThinkIMG/tool_server/tool_workers/scripts/launch_scripts"

commands=(
  "python ../../online_workers/controller.py --port 20001 --host 0.0.0.0"
  "python ../../online_workers/crop_worker.py --host 0.0.0.0 --port 20009 --no-register --controller-address http://127.0.0.1:20001"
  "python ../../online_workers/grounding_dino_worker.py --host 0.0.0.0 --port 20002 --no-register --controller-address http://127.0.0.1:20001"
  "python ../../online_workers/ocr_worker.py --host 0.0.0.0 --port 20008 --controller-address http://127.0.0.1:20001"
  "python ../../online_workers/zoomInsubfigure_worker.py --host 0.0.0.0 --port 20003 --controller-address http://127.0.0.1:20001"
  "python ../../online_workers/molmo_point_worker.py --host 0.0.0.0 --port 20005 --model_path allenai/Molmo-72B-0924 --controller-address http://127.0.0.1:20001"
  "python ../../online_workers/DrawVerticalLineByX_worker.py --host 0.0.0.0 --port 20006 --controller-address http://127.0.0.1:20001"
  "python ../../online_workers/DrawHorizontalLineByY_worker.py --host 0.0.0.0 --port 20007 --controller-address http://127.0.0.1:20001"
  "python ../../online_workers/grounding_dino_worker.py --host 0.0.0.0 --port 20002 --model-path ~/weights/groundingdino_swint_ogc.pth --model-config ~/weights/GroundingDINO_SwinT_OGC.py --controller-address http://127.0.0.1:20001"
)

for cmd in "${commands[@]}"; do
  # Extract exact filename after /online_workers/
  name=$(echo "$cmd" | grep -oP '(?<=/online_workers/)[^ ]+')
  
  echo "Starting $name..."
  screen -dmS "$name" bash -c "source .venv/bin/activate && cd $WORKDIR &&  $cmd; exec bash"
done

