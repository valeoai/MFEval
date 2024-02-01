# Towards Motion Forecasting with Real-World Perception Inputs: Are End-to-End Approaches Competitive? (ICRA 2024)
### [Paper](https://arxiv.org/abs/2306.09281) | [Webpage](https://valeoai.github.io/blog/publications/real-world-forecasting/)

### This is the official implementation of the evaluation protocol proposed in this work for motion forecasting models with real-world perception inputs.

[//]: # (## Getting Started)

[//]: # (- Installation)

[//]: # (- Prepare Dataset)

[//]: # (- Training and Evaluation)

##  Installation
#### Install dependencies
```bash
pip install numpy==1.23.5
pip install scipy==1.10.0
pip install nuscenes-devkit==1.1.10
```

## Prepare Dataset
#### Download nuScenes full dataset (v1.0) and map expansion (v1.3) [here](https://www.nuscenes.org/download).


#### Structure
After downloading, the dataset is put into data/ folder, the structure is as follows:
```
├── examples/
├── eval.py
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── v1.0-trainval/
│   │   ├── lidarseg/
```

##  Evaluation
Official benchmark of the nuScenes Prediction challenge is insufficient for evaluating motion forecasting deployed in real-world situations where the ground-truth inputs are not available. For this reason, we implement this evaluation library that allows any motion forecasting model with *real-world perception models* as inputs to benchmark their performance considering the imperfect nature of the forecasting inputs. It also provides a fair comparison between conventional and end-to-end methods. The evaluation protocol respects the guideline of the nuScenes Prediction challenge, which focuses on assessing the forecasting performance of *moving* vehicles.

### i. Prepare your results
A motion forecasting model with perception inputs should produce the following results:

```
All coordinates are in the !!GLOBAL!! frame defined in nuScenes dataset.
├── class_name: str, object class name, the valid value should be from ["car",  "truck", "bus"], see MAPPING_GT in eval.py. 
├── translation: list of float, the detection in bird's eye view (x,y) of shape (2) at t=0 (starting time step indicated by *sample token* in nuScenes dataset). 
├── detection_score: list of float, detection scores for "translation".
├── traj: np.array of float, future trajectories of shape [k, 12, 2]. k = #modes, "12" indicates that we evaluate 2Hz*6s frames of the future, "2" = (x,y) in BeV.
├── traj_prob: np.array of float, the associated confidence for each mode, of shape [k].
```

The results are saved in a json file with the following structure:
```
Example of JSON result file:
{
    sample_token_1: [
          {
            class_name: "car",
            translation: (900, 900)
            detection_score: int, ojbect detection score at t=0,
            traj: a float np.array of shape (k=10, 12, 2), future trajectory predictions.
            traj_prob: a float np.array of shape (k=10), future trajectory prediction confidences.
          },
          ...
         {
            class_name: "car",
            translation: (520, 520)
            detection_score: int, object detection score at t=0,
            traj: a float np.array of shape (k=10, 12, 2), future trajectory predictions.
            traj_prob: a float np.array of shape (k=10), future trajectory prediction confidences.
         },
    ],
    ...
    sample_token_n: [
        ...
    ]
}
```
### ii. Evaluate your results
```python
python eval.py --result_path=Path2YourResultJson --modes=NumModesToBeConsidered
```
The code will print the per-class minADE, minFDE, and APf, as well as the overall averaged performance. The metrics will be saved under the eval_results/ folder.

An example is given as follows:

#### You can download example json files [here](https://drive.google.com/drive/folders/1nSY253S2inR8MF3J51eLmtJ2mAV_J-SC?usp=sharing) and put them into the examples/ folder.
```python
python eval.py --result_path=./examples/lapred_voxelnext.json --modes=10
```
You should have:
```
{
    "car": {
        "Total_GT": 7180,
        "minADE": 1.664,
        "minFDE": 2.928,
        "MR_matched": 0.165,
        "class_name": "car",
        "best_score": 0.562,
        "mAPf": 0.389
    },
    "bus": {
        "Total_GT": 611,
        "minADE": 1.539,
        "minFDE": 2.465,
        "MR_matched": 0.113,
        "class_name": "bus",
        "best_score": 0.531,
        "mAPf": 0.478
    },
    "truck": {
        "Total_GT": 1203,
        "minADE": 1.815,
        "minFDE": 3.131,
        "MR_matched": 0.213,
        "class_name": "truck",
        "best_score": 0.442,
        "mAPf": 0.085
    },
    "all": {
        "num_modes": 10,
        "num_future_frames": 12,
        "Total_GT": 8994,
        "minADE": 1.669,
        "minFDE": 2.914,
        "MR_matched": 0.166,
        "mAPf": 0.317
    }
}
```

## License
This work is released under the [Apache 2.0 license](./LICENSE).

## Citation
If you find our work useful for your research, please consider citing the paper:
```bash
@inproceedings{xu2024realworldforecasting,
  title      = {Towards Motion Forecasting with Real-World Perception Inputs: Are End-to-End Approaches Competitive?},
  author     = {Yihong Xu and
                  Lo{\"{\i}}ck Chambon and
                  {\'{E}}loi Zablocki and
                  Micka{\"{e}}l Chen and
                  Alexandre Alahi and
                  Matthieu Cord and
                  Patrick P{\'{e}}rez},
  booktitle = {ICRA},
  year      = {2024}
}
```

## Acknowledgement

For the implementation of this evaluation library, we have got inspiration from the following work:

[1] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in CVPR, 2020.

[2] J. Gu, C. Hu, T. Zhang, X. Chen, Y. Wang, Y. Wang, and H. Zhao, “Vip3d: End-to-end visual trajectory prediction via 3d agent queries,” in CVPR, 2023.

[3] N. Peri, J. Luiten, M. Li, A. Osep, L. Leal-Taixé, and D. Ramanan, “Forecasting from lidar via future object detection,” in CVPR, 2022.


