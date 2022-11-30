import argparse
import os
import torch
import sys
import importlib
import numpy as np
from pathlib import Path
from data_loaders.ShapeNet import pc_normalize
from modules.utils import onehot, inplace_relu, fps, get_points_from_index
from data_loaders.constants import shapenetpart_cat2id, seg_classes


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='PointNetPP', help='model name')
    parser.add_argument('--device', type=str, default='cpu', help='specify device: cpu, mps, cuda:0')
    parser.add_argument('--n_point', type=int, default=2048, help='point Number')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--source', type=str, required=True, help="target point cloud")
    parser.add_argument('--weight', type=str, required=True, help="trained weight")
    parser.add_argument('--category', type=str, required=True, help="object class")
    parser.add_argument('--output_fn', type=str, default="output.txt", help="output file name")
    return parser.parse_args()


# python inference.py --device cpu --source ./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/04379243/1a8fe5baa2d4b5f7ee84261b3d20656.txt --category "Table" --use_normals  --weight ./log/PointNetPP2/checkpoints/best_model.pth


def main(args):

    base_dir = os.path.dirname(os.path.abspath(__file__))  # root dir
    sys.path.append(os.path.join(base_dir, 'models'))  # ./model

    num_parts = 50
    num_classes = 16

    # init model and load trained weights
    model_obj = importlib.import_module(args.model)
    segmentor = model_obj.PointNetPP(num_parts, use_normals=args.use_normals).to(args.device)
    checkpoint = torch.load(str(args.weight),map_location=torch.device(args.device))
    # dict_keys(['epoch', 'train_acc', 'test_acc', 'class_avg_iou',
    # 'instance_avg_iou', 'model_state_dict', 'optimizer_state_dict'])
    print("loaded weight at epoch:", checkpoint["epoch"])
    segmentor.load_state_dict(checkpoint['model_state_dict'])
    segmentor.apply(inplace_relu)

    # read source
    assert args.source[-4:] == ".txt", "Source file must be .txt format, split using spaces"
    data = np.loadtxt(args.source).astype(np.float32)[:, 0:6]
    if args.use_normals:
        # assert data.shape[1] == 6, "data should have six cols : x, y, z, Nx, Ny, Nz"
        point_set = data[:, 0:6]
    else:
        # assert data.shape[1] == 3, "data should have three cols : x, y, z"
        point_set = data[:, 0:3]
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

    assert args.category.lower() in shapenetpart_cat2id.keys(), "arg --category must belong to one of the classes: "\
                                                                + str(shapenetpart_cat2id.keys())

    # inference
    cls = torch.tensor(shapenetpart_cat2id[args.category.lower()], dtype=torch.long).to(args.device)
    points = torch.tensor(point_set, dtype=torch.float32).to(args.device).unsqueeze(0)
    # points = get_points_from_index(points, fps(points, args.n_point))
    points = points.transpose(2, 1)
    seg_pred = segmentor(points, onehot(cls, num_classes))
    pred = seg_pred.cpu().data.numpy()  # n_point 50
    logits = np.argmax(pred[:, :, seg_classes[args.category]], 2) + seg_classes[args.category][0]

    # write logits to file
    data = np.hstack((data, logits.transpose()))
    exp_dir = Path('./output/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = os.path.join(exp_dir, args.output_fn)
    np.savetxt(exp_dir, data, fmt='%.8e', delimiter=' ', newline='\n')
    print("inference finished, output saved to ", args.output_fn)


if __name__ == '__main__':
    args_ = parse_args()
    main(args_)
