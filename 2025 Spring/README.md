# Synthetic Data Generation and Evaluation Spring (and Winter) 2025

## Synthetic Data Generation with TableDiffusion (DPDM)

We use [TableDiffusion](https://gianluca.ai/table-diffusion/) by Gianluca Truda to generate synthetic data from the [eICU Collaborative Research Database](https://eicu-crd.mit.edu/). At current time, this is just a proof of concept. We do not guarantee privacy, and we only use a "demo" version of the eICU data that is [publicly available](https://physionet.org/content/eicu-crd-demo/2.0.1/).

### Setup

These steps need to be run once to set up this project for synthetic data generation.

1. Install dependencies. If you use apt: `sudo apt install curl git python-is-python3 python3 python3-dev python3-pip python3-venv`
2. Clone this repo: `git clone https://github.com/sinolyx1998/Synthetic-Data-Generation-Evaluation/`
3. Change to the "2025 Spring" directory: `cd "Synthetic-Data-Generation-Evaluation/Spring 2025"`
4. Set up and activate a python virtual environment: `python3 -m venv . && source bin/activate`
5. (Optional) If you are not using CUDA, first install pytorch without CUDA support. It will make the next step go faster. `pip install torch --index-url https://download.pytorch.org/whl/cpu`
6. Install python dependencies: `pip install -r requirements.txt`
7. Clone the TableDiffusion code to the modules directory: `mkdir -p modules && git clone https://github.com/gianlucatruda/TableDiffusion.git modules/TableDiffusion`
8. Download and extract the demo patient.csv file: `curl -LO "https://physionet.org/files/eicu-crd-demo/2.0.1/patient.csv.gz?download" && gunzip patient.csv.gz`

### Generating Synthetic Data

1. Change to the "2025 Spring" directory if you are not already there: `cd "Synthetic-Data-Generation-Evaluation/Spring 2025"`
2. Run `python generate-dpdm.py` with whatever arguments you want. Options are specified below:

```
--epsilon EPSILON       Specify epsilon (default is 1.0)
--delta DELTA           Specify delta (default is 10^-5)
--cuda BOOL             Whether to use CUDA (default is False)
-i, --input FILE        Input file (default is patient.csv)
-o, --output FILE       Output file (default is synthetic_data.csv)
-h, --help              Show this screen
```

## License

GPLv3
