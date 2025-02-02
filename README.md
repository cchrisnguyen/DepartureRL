# Fuel- and Noise-Minimal Departure Trajectory using Deep Reinforcement Learning with Aircraft Dynamics and Topography Constraints

This repository contains the code for the paper titled "Fuel- and noise-minimal departure trajectory using deep reinforcement learning with aircraft dynamics and topography constraints" in press at Communications in Transportation Research.

## Requirements

To set up the environment, ensure you have Python version >3.10 installed. You can create a Conda environment and install the required packages using the following steps:

1. **Create a Conda environment:**

    ```sh
    conda create -n departure_rl python=3.10
    ```

2. **Activate the Conda environment:**

    ```sh
    conda activate departure_rl
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

## Training

Before training, you need to apply for a BADA license at [EUROCONTROL BADA](https://www.eurocontrol.int/model/bada) and place the BADA files into `./env/performance/BADA/`. Additionally, download the population and terrain data from [this link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/hcnguyenaa_connect_ust_hk/EqTiJTiS_tBFlyeS_dAHofEBKHlRlFDME-W5cVnV_3MPRw?e=rfPaaI) and put them in `./env/topography_data/`.

To train the model, use the following command:
```sh
python -W ignore main.py
```

## Inference

To run inference, use the following command:
```sh
python main.py --render True --model_index -1 
```

## Citation
If you use this code in your research, please cite the papers:

```
@article{nguyen2025fuel,
  title={Fuel- and noise-minimal departure trajectory using deep reinforcement learning with aircraft dynamics and topography constraints},
  author={Nguyen, Chris H and Shihua, James M and Liem, Rhea P},
  journal={Communications in Transportation Research},
  year="\noop{3001}in press",
  publisher={Elsevier}
}
```
and
```
@inproceedings{nguyen2023noise,
  title={Noise-and Fuel-Minimal Departure Trajectory Optimization with Reinforcement Learning},
  author={Nguyen, Chris H and Shihua, James M and Hui, Ka Yiu and Liem, Rhea P},
  booktitle={AIAA AVIATION 2023 Forum},
  pages={4215},
  year={2023}
}
```
