import os
import sys
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
from os.path import join as pjoin
import argparse
import importlib

from data_loader import process_single_bvh, process_single_json

from trainer import Trainer
from remove_fs import remove_fs, save_bvh_from_network_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--content_src', type=str, default=None)
    parser.add_argument('--style_1', type=str, default=None)
    parser.add_argument('--style_2', type=str, default=None)
    parser.add_argument('--proportion', type=float, default=None)
    parser.add_argument('--output_dir', type=str, default=None)

    return parser.parse_args()


def main(args):
    config_module = importlib.import_module(args.config)
    config = config_module.Config()

    # Load experiment setting
    config.initialize(args)

    # Trainer
    trainer = Trainer(config)
    trainer.to(config.device)
    trainer.resume()

    co_data = process_single_bvh(args.content_src, config, to_batch=True)
    cl_b = process_single_bvh(args.style_1, config, to_batch=True)
    cl_c = process_single_bvh(args.style_2, config, to_batch=True)
    proportion = args.proportion

    output = trainer.test_interpolate(co_data, cl_b, cl_c, proportion)
    foot_contact = output["foot_contact"][0].cpu().numpy()
    motion = output["trans"][0].detach().cpu().numpy()
    output_dir = pjoin(config.main_dir, 'test_output') if args.output_dir is None else args.output_dir
    save_bvh_from_network_output(motion, output_path=pjoin(output_dir, 'raw.bvh'))
    remove_fs(motion, foot_contact, output_path=pjoin(output_dir, 'fixed.bvh'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
