# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import paddle
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from ModelNetDataset import ModelNetDataset
from models.pointnet_plus_plus_ssg import Pointnet2_cls_ssg
from paddle.optimizer import Adam
import paddle.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores with voting')
    return parser.parse_args()

def test(model, loader, num_class=40, vote_num=1):

    mean_correct = []

    model.eval()
    
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        # points, target = points.cuda(), target.cuda()

        points = points.transpose((0, 2, 1))
        
        pred, _ = model(points)
        pred_choice = paddle.argmax(pred, axis=1)
        
        correct = pred_choice.equal(target).astype("float32").sum()
        mean_correct.append(correct.numpy()[0] / float(points.shape[0]))

    instance_acc = np.mean(mean_correct)

    return instance_acc

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'dataset/modelnet40_normal_resampled/'

    test_dataset = ModelNetDataset(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = paddle.io.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]

    classifier = Pointnet2_cls_ssg(num_class, normal_channel=args.use_normals)

    checkpoint = paddle.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.set_state_dict(checkpoint['model_state_dict'])

    with paddle.no_grad():
        instance_acc = test(classifier, testDataLoader, vote_num=args.num_votes, num_class=num_class)
        log_string('Test Instance Accuracy: %f' % (instance_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)