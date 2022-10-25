import argparse
from multiprocessing import dummy
import os
from loguru import logger

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module
from yolox.evaluators.coco_evaluator import COCOEvaluator

from patch_gpu_to_cpu import patch_cuda

from bigdl.nano.pytorch import InferenceOptimizer


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--decode_in_inference",
        action="store_true",
        help="decode in inference or not"
    )

    return parser

@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    model.eval()

    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt
    # load the model state dict
    
    ckpt = torch.load(ckpt_file, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt["model"])
    logger.info("loading checkpoint done.")

    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1], requires_grad=False)
    # preds = model(dummy_input).detach()
    # train_set = TensorDataset(dummy_input, preds)
    train_set = TensorDataset(dummy_input)
    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    # train_loader = exp.get_data_loader(args.batch_size, False)
    coco_evaluator = exp.get_evaluator(args.batch_size, False)
    coco_evaluator.per_class_AP = True
    coco_evaluator.per_class_AR = True

    def metric(model, data=None):
        with torch.no_grad():
            ap50_95, ap50, _ = coco_evaluator.evaluate(model, False)
            return ap50

    inference_optimizer = InferenceOptimizer()
    inference_optimizer.optimize(model, metric=metric, training_data=train_loader)
    # inference_optimizer.trace(model, accelerator='jit', input_sample=dummy_input)
    # inference_optimizer.quantize(model, precision = 'int8')
    inference_optimizer.summary()

if __name__ == "__main__":
    patch_cuda()
    main()