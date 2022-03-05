ssh lwong@txe1-login.mit.edu

scp setup* lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion
scp Makefile lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion
scp MANIFEST.in lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion
scp requirements.txt lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion
scp orion/benchmark.py lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion/orion/benchmark.py
scp orion/data/parameters.csv lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion/orion/data/parameters.csv
scp orion/data/datasets.csv lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion/orion/data/datasets.csv

scp *.sh lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion
scp experiments.py lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion
scp orion/*.py lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion/orion/
scp orion/primitives/*.py lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion/orion/primitives/
scp orion/primitives/jsons/*.json lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion/orion/primitives/jsons
scp -r orion/pipelines lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion/orion/

scp orion/primitives/jsons/orion.primitives.tadgan.TadGAN.json lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion/orion/primitives/jsons
scp orion/primitives/tadgan.py lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion_MultiGAN/orion/primitives/scp orion/primitives/attention_layers.py lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion/orion/primitives/
rsync -a --ignore-existing lwong@txe1-login.mit.edu:/home/gridsan/lwong/Orion_1.0/notebooks/* /Volumes/easystore/thesis/results

LLstat
LLsub run.sh
LLkill

conda install cudatoolkit==10.1
