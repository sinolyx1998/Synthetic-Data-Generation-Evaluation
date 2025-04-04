#!/bin/sh

sudo apt install curl git python-is-python3 python3{,-dev,-pip,-venv}
python3 -m venv . && source bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
mkdir -p modules && git clone https://github.com/gianlucatruda/TableDiffusion.git modules/TableDiffusion
curl -LO "https://physionet.org/files/eicu-crd-demo/2.0.1/patient.csv.gz?download" && gunzip patient.csv.gz
mv patient.csv patient.csv.orig && sed '/,,/d' patient.csv.orig > patient.csv
