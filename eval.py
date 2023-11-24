# Copyright 2023 - Valeo Comfort and Driving Assistance - Yihong Xu @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
from typing import List, Dict, Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# for filtering sample ids with nuscene split
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes import NuScenes
import copy

# set seeds #
import random
random.seed(0)
np.random.seed(0)

class cfg:
    nelem = 101 # number of interpolations for recall and precision plot
    pred_traj_num = 10 # number of modes 
    past_frame_num = 4 # number of past trajs to be considered 
    future_frame_num = 12 # number of future trajs
    max_dis_from_ego = 50.0 # not used, max distance from the ego car
    matching_threshold = 2.0 # meters, the threshold deciding if a prediction at t=0 could be a match with a gt at t=0 
    miss_rate_threshold = 4.0 # meters, the maximum FDE, beyond this distance, it is considered as a traj FP, also the gt traj is considered miss.
    min_recall = 0.1
    min_precision = 0.1
    


MAPPING_PRED = {
    "car": "car",
    "construction_vehicle": "construction_vehicle",
    "truck": "truck",
    "bus": "bus",
    'vehicle.emergency.police': 'vehicle.emergency.police',
    "trailer": "trailer",
    "barrier": "barrier",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "pedestrian": "pedestrian",
    "traffic_cone": "traffic_cone"
}

MAPPING_GT = {
    'vehicle.bus.rigid': "bus",
    'vehicle.car': "car",
    'vehicle.truck': "truck",
    'vehicle.bus.bendy': "bus",
    'vehicle.construction': "construction_vehicle",
    'vehicle.emergency.police': 'vehicle.emergency.police',

    'movable_object.barrier': "barrier",
    'vehicle.bicycle': "bicycle",
    'vehicle.motorcycle': "motorcycle",
    'human.pedestrian.adult': "pedestrian",
    'human.pedestrian.child': "pedestrian",
    'human.pedestrian.construction_worker': "pedestrian",
    'human.pedestrian.police_officer': "pedestrian",
    'movable_object.trafficcone': "traffic_cone",
    'vehicle.trailer': "trailer"

}


class Metric:
    def __init__(self):
        self.values = []

    def accumulate(self, value):
        if value is not None:
            self.values.append(value)

    def get_mean(self):
        if len(self.values) > 0:
            return np.mean(self.values)
        else:
            return 0.0

    def get_sum(self):
        return np.sum(self.values)


class PredictionMetrics:
    def __init__(self):
        self.minADE = Metric()
        self.minFDE = Metric()
        self.gt_agent_num = Metric()
        self.MR_matched = Metric()

    def serialize(self) -> Dict[str, Any]:

        return dict(
            Total_GT=int(self.gt_agent_num.get_sum()),
            # metrics for matched objects
            minADE= round(self.minADE.get_mean(),3) if len(self.minADE.values)>0 else None,
            minFDE= round(self.minFDE.get_mean(),3) if len(self.minFDE.values)>0 else None,
            MR_matched = round(self.MR_matched.get_mean(),3) if len(self.MR_matched.values)>0 else None
            
        )

# class for one GT agent #
class GTAgent:
    def __init__(self,
                 # t=0
                 translation: np.ndarray = np.zeros(2),

                 # future
                 traj: np.ndarray = np.zeros((cfg.future_frame_num, 2)),
                 traj_is_valid: np.ndarray = np.zeros(cfg.future_frame_num, dtype=np.float32),

                 # past
                 past_traj: np.ndarray = np.zeros((cfg.past_frame_num, 2)),
                 past_traj_is_valid: np.ndarray = np.zeros(cfg.past_frame_num, dtype=np.float32),

                 # meta
                 instance_token: str = '',
                 class_name = '',
                 ):
        self.translation = translation.copy()
        self.traj = traj.copy()
        self.traj_is_valid = traj_is_valid.copy()
        self.past_traj = past_traj.copy()
        self.past_traj_is_valid = past_traj_is_valid.copy()
        self.instance_token = instance_token
        self.class_name = class_name
        assert self.class_name != '', 'class_name cannot be empty.'


# class for one predicted agent #
class PredAgent:
    def __init__(self,
                 # meta
                 sample_token: str = '',
                 class_name: str = '',

                 # t=0
                 translation: np.ndarray = np.zeros(2),
                 detection_score: np.ndarray = np.zeros(1),
                 
                 # future of multiple modes
                 traj: np.ndarray = np.zeros((cfg.pred_traj_num, cfg.future_frame_num, 2)),
                 traj_prob: np.ndarray = np.zeros(cfg.pred_traj_num)
                 ):
        # meta         
        self.sample_token = sample_token
        self.class_name = class_name
        assert self.class_name != '', 'class_name cannot be empty.'

        # t=0
        self.translation = translation.copy()
        self.detection_score = detection_score

        # future
        assert cfg.pred_traj_num <= traj.shape[0]
        if cfg.pred_traj_num <  traj.shape[0]:
            topk_indx = np.argsort(traj_prob, axis=0)[::-1][:cfg.pred_traj_num]
            self.traj = traj[topk_indx]
            self.traj_prob = traj_prob[topk_indx]
        else:
            self.traj = traj
            self.traj_prob = traj_prob



    @classmethod
    def deserialize(cls, content: dict):
        """ Load information from result json file."""
        return cls(
                 # meta
                 sample_token = content['sample_token'],
                 class_name = content['class_name'],

                 # t=0
                 translation = np.array(content['translation'][:2]),
                 detection_score = np.array(content['detection_score']),
                 
                 # future traj. of multiple modes
                 traj = np.array(content['traj']),
                 traj_prob = np.zeros(cfg.pred_traj_num) if 'traj_prob' not in content.keys() or content['traj_prob'] is None else np.array(content['traj_prob']),

        )
        

class PredictionEval:
    def __init__(self,
                 result_path: str = None,
                 output_dir: str = None,
                 nuscenes=None,
                 helper=None):
        """
        Parameters
        ----------
        :param result_path: Path of the JSON result file.
        :param output_dir: Folder to save metrics.
        :param nuscenes: nuScenes toolkit.
        :param helper: nuScenes Prediction helper.
        """

        """
        Example of JSON result file:
        {
            sample_token_1: [
                    {
                    class_name: string, object class name, should be from ["car", "truck", "bus"] 
                    translation: float np.array of shape (2), object location at t=0 in BEV (x,y),
                    detection_score: float, ojbect detection score at t=0,
                    traj: float np.array of shape (cfg.pred_traj_num, cfg.future_frame_num, 2), future trajectory predictions,
                    traj_prob: float np.array of shape (cfg.pred_traj_num), future trajectory prediction confidences.
                    },
                ...
                {
                    class_name: string, object class name, should be from ["car", "truck", "bus"] 
                    translation: float np.array of shape (2), object location at t=0 in BEV (x,y),
                    detection_score: float, ojbect detection score at t=0,
                    traj: float np.array of shape (cfg.pred_traj_num, cfg.future_frame_num, 2), future trajectory predictions,
                    traj_prob: float np.array of shape (cfg.pred_traj_num), future trajectory prediction confidences.
                },
            ],
            ...
            sample_token_n: [
                ...
            ]
        }
        """
        self.result_path = result_path

        if output_dir is None:
            # set to the directory of `result_path`
            output_dir = os.path.split(result_path)[0]

        self.output_dir = output_dir

        # Check result file exists.
        assert os.path.exists(result_path), f'Error: The result file {result_path} does not exist!'

        # Make dirs.
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.kept_classes = ['vehicle.bus.rigid', 'vehicle.car', 'vehicle.truck', 'vehicle.bus.bendy',  'vehicle.construction',  'vehicle.emergency.police']

        self.nuscenes = nuscenes
        self.helper = helper
        instance_sample = get_prediction_challenge_split("val", dataroot="./data/nuscenes/") # {instance_token}_{sample_token}
        self.sample_tokens = sorted(list(set([x.split("_")[1] for x in instance_sample])))
        self.all_instances = instance_sample
        print("Valid #instances in nuScenes Prediction: ", len(instance_sample))
        print("Valid #frames (samples) in nuScenes Prediction: ",  len(self.sample_tokens))

        # load prediction #
        with open(result_path) as f:
            print("evaluating: ", result_path)
            print("Warning: please make sure that your predicition result has filtered out classes such as " +
            " ['movable_object.barrier', 'vehicle.bicycle', 'vehicle.motorcycle', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker', 'human.pedestrian.police_officer', 'movable_object.trafficcone', 'vehicle.trailer']. " )

            
            data = json.load(f)
            if 'results' in data:
                data = data['results']

            self.sample_token_2_pred_agents = {}
            for sample_token, boxes in data.items():
                
                if sample_token in self.sample_tokens: # xyh only those in NuScenes Prediciton split
                    self.sample_token_2_pred_agents[sample_token] = [PredAgent.deserialize(each) for each in boxes]


        self.metrics = []
    
    def traj_fde(self, gt_box, pred_box, final_step):
        if gt_box.traj.shape[0] <= 0:
            return np.inf
        final_step = min(int(gt_box.traj_is_valid.sum()), final_step)
        gt_final = gt_box.traj[None, final_step-1]
        pred_final = np.array(pred_box.traj)[:,final_step-1,:]
        err = gt_final - pred_final
        err = np.sqrt(np.sum(np.square(gt_final - pred_final), axis=-1))
        return np.min(err)

    def evaluate(self):

        # To avoid penalizing detections as FP that match with gts outside valid gts
        # we match predictions at this sample token with ALL gts and filter those matched with gts-valid_gts
        new_sample_token_2_pred_agents = {}
        print("Filtering out Preds matched with non valid GTs...")
        for index in tqdm(range(len(self.sample_tokens))):
            sample_token = self.sample_tokens[index]
            new_sample_token_2_pred_agents[sample_token] = []
            if sample_token not in self.sample_token_2_pred_agents.keys():
                continue
            preds_this_token = self.sample_token_2_pred_agents[sample_token]
            all_gt_agents = get_gt_agents(sample_token, self.helper, self.kept_classes, self.all_instances, only_valid=False)

            
            if len(all_gt_agents) > 0:
                cost_matrix = np.zeros((len(preds_this_token), len(all_gt_agents)))
                gt_translations = np.array([each.translation for each in all_gt_agents])

                # dist per prediction to all gts
                for i in range(len(preds_this_token)):
                    cost_matrix[i] = get_distances(preds_this_token[i].translation, gt_translations)
                    # at t=0, do not assign to those that are far away.
                    cost_matrix[i][np.nonzero(cost_matrix[i] > cfg.matching_threshold)] = 10000.0

                pred_list, gt_list = linear_sum_assignment(cost_matrix)

                assert len(np.unique(pred_list)) == len(pred_list)
                
                # I want [not_matched, matched and is with valid gt]
                for pred_i in range(len(preds_this_token)):
                    if pred_i in pred_list:
                        i = pred_list.tolist().index(pred_i)
                        if cost_matrix[pred_i, gt_list[i]] <= cfg.matching_threshold: # matched
                            if all_gt_agents[gt_list[i]].instance_token+'_'+sample_token in self.all_instances: # is valid
                                new_sample_token_2_pred_agents[sample_token].append(preds_this_token[pred_i])
                        else: # not matched because of threshold
                            new_sample_token_2_pred_agents[sample_token].append(preds_this_token[pred_i])
                    else: # not matched
                        new_sample_token_2_pred_agents[sample_token].append(preds_this_token[pred_i])
            else: # no gt => no matched
                new_sample_token_2_pred_agents[sample_token] = copy.deepcopy(self.sample_token_2_pred_agents[sample_token])

        self.sample_token_2_pred_agents = new_sample_token_2_pred_agents
        
        aps = []
        score_ths = []

        all_metrics = []
        classes = ["car", "bus", "truck"]
        acc_all_metrics = PredictionMetrics()

        # calculate AP for traj, reflecting the detection quality and forecasting quality
        for class_name in classes:

            print(f"**********************Evaluating {class_name}**********************")
            kept_classes = []
            for k, v in MAPPING_GT.items():
                if v == class_name:
                    kept_classes.append(k)
            
            
            # collect all gts, all preds from all samples
            all_gts = {}
            all_preds = []
            all_pred_scores = []
            # count the positives
            npos = 0            

            print(f"Collecting GTs and Preds from all samples (frames) of class {class_name}...")
            for index in tqdm(range(len(self.sample_tokens))):
                sample_token = self.sample_tokens[index]
                # the class filtering is included inside #
                gts = get_gt_agents(sample_token, self.helper, kept_classes, self.all_instances, only_valid=True)
                all_gts[sample_token] = gts
                npos += len(gts)


                if sample_token not in self.sample_token_2_pred_agents:
                    pass
                else:
                    cur_preds = []
                    cur_pred_scores = []
                    
                    for pred in self.sample_token_2_pred_agents[sample_token]:
                        # filter by class
                        if MAPPING_PRED[pred.class_name] == class_name:
                            cur_preds.append(pred)
                            cur_pred_scores.append(pred.detection_score)

                    all_preds += cur_preds
                    all_pred_scores += cur_pred_scores
            print("Found {} GT of class {} across {} samples.".
              format(npos, class_name, len(self.sample_tokens)))

            print("Found {} PRED of class {} across {} samples.".
              format(len(all_pred_scores), class_name, len(self.sample_tokens)))
            
            assert len(all_pred_scores) == len(all_preds)

            if npos == 0 or len(all_pred_scores) ==0:
                recall = np.zeros(cfg.nelem)
                precision = np.zeros(cfg.nelem)
                f1 = np.zeros(cfg.nelem)
                conf=np.zeros(cfg.nelem)
                # score threshold with best f1 score 
                score_th = conf[np.argmax(f1)]

            else:
                
                # get recall and precision for all sample tokens
                # Sort by confidence.
                sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(all_pred_scores))][::-1]
                
                # Do the actual matching.
                tp = []  # Accumulator of true positives.
                fp = []  # Accumulator of false positives.
                conf = []  # Accumulator of confidences.
                taken = set()  # Initially no gt bounding box is matched.

                
                for ind in sortind: # from high scores to low scores, = thresholding, for loop all preds
                    pred_box = all_preds[ind]
                    min_dist = np.inf
                    match_gt_idx = None
                    fde_distance = np.inf
                    
                    # matched with all gts => determine fp
                    for gt_idx, gt_box in enumerate(all_gts[pred_box.sample_token]): #for loop all gts
                        # Find closest match at t=0 among ground truth boxes
                        if MAPPING_GT[gt_box.class_name] == class_name and not (pred_box.sample_token, gt_idx) in taken:
                            this_distance = np.linalg.norm(gt_box.translation - pred_box.translation, axis=-1)
                            if this_distance < min_dist:
                                min_dist = this_distance
                                match_gt_idx = gt_idx
                                fde_distance = self.traj_fde(gt_box, pred_box, cfg.future_frame_num)
                    
                    # If the closest match is close enough according to threshold we have a match! 
                    is_match = min_dist < cfg.matching_threshold and fde_distance < cfg.miss_rate_threshold

                    assert pred_box.detection_score == all_pred_scores[ind]
                    conf.append(float(pred_box.detection_score))

                    if is_match:
                        taken.add((pred_box.sample_token, match_gt_idx))
                        tp.append(1)
                        fp.append(0)
                    else:
                        tp.append(0)
                        fp.append(1)

                     
                
                # Accumulate.
                tp = np.cumsum(tp).astype(float)
                fp = np.cumsum(fp).astype(float)
                conf = np.array(conf)
                
            
                # Calculate precision and recall.
                prec = tp / (fp + tp)
                rec = tp / float(npos)
                f1 = 2 * prec * rec /(prec+rec+1e-5)

                # score threshold with best f1 score 
                score_th = conf[np.argmax(f1)]

                rec_interp = np.linspace(0, 1, cfg.nelem)  # 101 steps, from 0% to 100% recall.
                prec = np.interp(rec_interp, rec, prec, right=0)
                conf = np.interp(rec_interp, rec, conf, right=0)
                rec = rec_interp
                precision=prec

            
            #Calculate AP, based on PR curve for this class
            ap = calc_ap(precision,
                    cfg.min_recall,
                    cfg.min_precision)
            
            
            # collect ap; best_score_th (based on best f1 score), recall, precision of this th.
            aps.append(ap)
            score_ths.append(score_th)


            print(f"Best score threshold for {class_name}:", round(score_th, 3))
            print(f"APf for {class_name}: ", round(ap, 3))

            # After filtering prediction results based on best F1 score, calculate FDE, ADE, MR for the matched objects.
            # these only reflect the performance of MATCHED objects.
            metrics = PredictionMetrics()
            print("Evaluating minADE, minFDE, MR with the best threshold...")
            for index in tqdm(range(len(self.sample_tokens))):
                sample_token = self.sample_tokens[index]
                # only_valid is set to True because you already filter those matched with non valid gt.
                gt_agents: List[GTAgent] = get_gt_agents(self.sample_tokens[index], self.helper, kept_classes, self.all_instances, only_valid=True)
                    
                    
                pred_agents: List[PredAgent] = copy.deepcopy(self.sample_token_2_pred_agents[sample_token])

                # filter by class and score th:
                cur_preds = []
                for pred in pred_agents:
                    if MAPPING_PRED[pred.class_name] == class_name and pred.detection_score >= score_th:
                        cur_preds.append(pred)
                pred_agents = cur_preds

                if len(gt_agents) > 0:
                    matched_of_gt_box = np.ones(len(gt_agents), dtype=np.int32) * -1
                    cost_matrix = np.zeros((len(pred_agents), len(gt_agents)))
                    gt_translations = np.array([each.translation for each in gt_agents])

                    # dist per prediction to all gts
                    for i in range(len(pred_agents)):
                        cost_matrix[i] = get_distances(pred_agents[i].translation, gt_translations)
                        # at t=0, do not assign to those that are far away.
                        cost_matrix[i][np.nonzero(cost_matrix[i] > cfg.matching_threshold)] = 10000.0

                    r_list, c_list = linear_sum_assignment(cost_matrix)

                    for i in range(len(r_list)):
                        if cost_matrix[r_list[i], c_list[i]] <= cfg.matching_threshold:
                            matched_of_gt_box[c_list[i]] = r_list[i]


                    for i in range(len(gt_agents)):
                        box_idx = matched_of_gt_box[i]
                        gt_agent = gt_agents[i]

                        if box_idx == -1: # no pred is assigned to this gt.
                            minADE = None
                            minFDE = None
                            MR_matched = None
                        else:
                        
                            pred_agent = pred_agents[box_idx]
                            argmin, minADE, minFDE = get_argmin_trajectory(gt_agent.traj, gt_agent.traj_is_valid, pred_agent.traj)

                            assert minFDE is not None
                            MR_matched = minFDE > cfg.miss_rate_threshold

                        metrics.minADE.accumulate(minADE)
                        metrics.minFDE.accumulate(minFDE)
                        metrics.MR_matched.accumulate(MR_matched)

                    metrics.gt_agent_num.accumulate(len(gt_agents))
            
            print("********************************************************************")
            print()
            all_metrics.append(metrics)
            acc_all_metrics.minADE.values += metrics.minADE.values
            acc_all_metrics.minFDE.values += metrics.minFDE.values
            acc_all_metrics.MR_matched.values += metrics.MR_matched.values
            acc_all_metrics.gt_agent_num.values += metrics.gt_agent_num.values

        return all_metrics, acc_all_metrics, aps, classes, score_ths

    def main(self) -> Dict[str, Any]:
        # evaluate #
        all_metrics, acc_all_metrics, aps, classes, score_ths = self.evaluate()
        
        
        
        # show and save results #
        all_metrics_summary = {}
        
        for metrics, ap, class_name, best_score_th in zip(all_metrics, aps, classes, score_ths):
            metrics_summary = metrics.serialize()
            metrics_summary["class_name"] = class_name
            metrics_summary["best_score"] = round(float(best_score_th), 3)
            metrics_summary["mAPf"] = round(ap, 3)
            all_metrics_summary[class_name] = metrics_summary
            
        all_metrics_summary['all'] = {
            "num_modes": int(cfg.pred_traj_num),
            "num_future_frames": int(cfg.future_frame_num),
            'Total_GT': int(acc_all_metrics.gt_agent_num.get_sum()),
            'minADE': round(float(acc_all_metrics.minADE.get_mean()), 3),
            'minFDE': round(float(acc_all_metrics.minFDE.get_mean()), 3),
            'MR_matched': round(float(acc_all_metrics.MR_matched.get_mean()),3),
            'mAPf': round(float(np.mean(aps)), 3)
        }
        
        res_name = self.result_path.split("/")[-1]
        os.makedirs(self.output_dir + '/eval_results/', exist_ok=True)
        with open(f'{self.output_dir}/eval_results/K={cfg.pred_traj_num}_{res_name}', 'w') as f:
            json.dump(all_metrics_summary, f, indent=4)
        print(json.dumps(all_metrics_summary, indent=4))
        print(f'Results saved to: {self.output_dir}/eval_results/K={cfg.pred_traj_num}_{res_name}')

        return all_metrics_summary


# Functions for distance calculation #
def get_distances(point, points):
    assert point.ndim == 1 and points.ndim == 2
    return np.sqrt(np.square(points[:, 0] - point[0]) + np.square(points[:, 1] - point[1]))

def get_distances_norm(point, pred_ava, points, ava):
    assert point.ndim == 2 and points.ndim == 3
    return np.linalg.norm((point[None, :, :] - points)*ava[:,:, None]*pred_ava[None, :, None], axis=-1).mean(-1)


def get_argmin_trajectory(future_traj, future_traj_is_valid, pred_future_trajs):
    if future_traj_is_valid.sum() == 0:
        print("You'll never enter here because all valid gts have full future path.")
        return None, None, None

    delta: np.ndarray = pred_future_trajs - future_traj[np.newaxis, :]
    assert delta.shape == (cfg.pred_traj_num, cfg.future_frame_num, 2)

    delta = np.sqrt((delta * delta).sum(-1))
    assert delta.shape == (cfg.pred_traj_num, cfg.future_frame_num)

    if future_traj_is_valid[-1]:
        minFDE = delta[:, -1].min()
    else:
        print("You'll never enter here because all valid gts have full future path.")
        minFDE = None

    delta = delta * future_traj_is_valid[np.newaxis, :]
    delta = delta.sum(-1) / future_traj_is_valid.sum()
    assert delta.shape == (cfg.pred_traj_num,)

    argmin = delta.argmin()
    minADE = delta.min()

    return argmin, minADE, minFDE

# Function for gt annotation collection #
def get_gt_agents(sample_token, helper, kept_classes, all_instances, only_valid):
    # collect gt annotations by helper #
    gt_agents_pred = []

    future_just_xy_sample = helper.get_future_for_sample(sample_token, seconds=cfg.future_frame_num//2, in_agent_frame=False, just_xy=True)  # "agent token: xy [12,2] in global frame"        
    future_xy_sample = helper.get_future_for_sample(sample_token, seconds=cfg.future_frame_num//2, in_agent_frame=False, just_xy=False)  # "agent token: xy [12,2] in global frame"
    past_just_xy_sample = helper.get_past_for_sample(sample_token, seconds=cfg.past_frame_num//2, in_agent_frame=False, just_xy=True)  # "agent token: xy [4,2] in global frame"        
    
    for instance_token, v in future_just_xy_sample.items(): # iterate through agents
        instance_info = future_xy_sample[instance_token]
        v_past = past_just_xy_sample[instance_token]
        if (len(instance_info) == 0 or instance_info[0]['category_name'] not in kept_classes): # not the category of interest or no info
                continue

        # only valid, nuScene only evaluate a subset of objects present at t=0
        if only_valid and instance_token+'_'+sample_token not in all_instances:
            continue
        
        # get t zero
        t_zero_annotation = helper.get_sample_annotation(instance_token, sample_token)

        # get past
        gt_xy_past = np.zeros(shape=(cfg.past_frame_num-1, 2), dtype=np.float32) # [3, 2=(x,y)]
        gt_availability_past = np.zeros(shape=(cfg.past_frame_num-1), dtype=np.float32) # [1, 3]
        if v_past.shape[0] > 3:
            v_past = v_past[:3] # we only need [frame -1, frame -2, frame -3]
        if v_past.shape[0] > 0:
            gt_availability_past[:v_past.shape[0]] += 1
            gt_xy_past[:v_past.shape[0], :] = copy.deepcopy(v_past.astype(np.float32))
        # order to [-3, -2, -1]
        gt_availability_past = gt_availability_past[::-1]
        gt_xy_past = gt_xy_past[::-1, :]
        
        # get past 
        gt_past_full = np.concatenate([gt_xy_past, np.array(t_zero_annotation['translation'])[np.newaxis, :2]], axis=0)
        gt_past_ava_full = np.concatenate([gt_availability_past, np.ones(shape=(1))], axis=0)


        # get future
        gt_availability = np.zeros(shape=(cfg.future_frame_num))
        gt_availability[:v.shape[0]] = 1
        gt_full_length = np.zeros(shape=(cfg.future_frame_num, 2))
        gt_full_length[:v.shape[0], :] = copy.deepcopy(v.astype(np.float32))

        gt_agent_pred = GTAgent(
                instance_token = instance_token,
                translation = np.array(t_zero_annotation['translation'][:2]),
                past_traj = gt_past_full.copy(),
                past_traj_is_valid = gt_past_ava_full.copy(),
                traj = gt_full_length,
                traj_is_valid = gt_availability,
                class_name = t_zero_annotation['category_name']

        )

        gt_agents_pred.append(gt_agent_pred)
    return gt_agents_pred

# function for Average Precision Calculation #
def calc_ap(precision : np.ndarray, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)

def main():
    # args #
    parser = argparse.ArgumentParser(description='Prediction evaluation')
    parser.add_argument('--result_path', default='', help='path to prediction results in JSON format')
    parser.add_argument('--modes', default=10, type=int, help='Number of modes in future trajectories to be considered.')
    args = parser.parse_args()
    
    # update modes in cfg #
    cfg.pred_traj_num = args.modes

    # load nuScenes toolkit #
    nuscenes = NuScenes('v1.0-trainval/', dataroot="./data/nuscenes/")
    helper = PredictHelper(nuscenes)
    
    # nuScenes eval #
    nusc_eval = PredictionEval(
                                result_path=args.result_path,
                                nuscenes=nuscenes,
                                helper=helper)
    nusc_eval.main()

if __name__ == '__main__':
    main()

