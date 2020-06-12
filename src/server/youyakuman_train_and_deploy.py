import random
import torch
import os
import json
import glob
from io import BytesIO
import numpy as np

import argparse
from configparser import ConfigParser

# Training class
from lib.models.train_dataloader import DataLoader
from lib.models.model_builder import Summarizer, SummarizerParams, build_optim
from lib.models.trainer import build_trainer

# Inference class
from lib.ModelExecutor import ModelExecutor
from lib.Preprocessor import Preprocessor, InferInputType

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
sys.path.append(os.path.join(os.path.dirname(__file__), "lib", "models"))

def model_fn(model_dir):
    base_path = "/opt/ml/model/code"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load model dict
    model_path = get_model_path(model_dir)
    model_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_params = SummarizerParams.from_args(model_state_dict['opt'])
    model_params.device = device
    # update pretrained_model_path
    # TODO : should do to upload pretrained model data into S3 or save model folder with fine-tuning model
    base_pretrained_model_path = model_state_dict['opt'].pretrained_model_path
    pretrained_model_path = os.path.join(base_path, base_pretrained_model_path)
    model_params.pretrained_model_path = pretrained_model_path
    
    # create model
    model = Summarizer(params=model_params)
    model.load_cp(pt=model_state_dict['model'])
    
    # get preprocessor to execute inference
    config_path = os.path.join(base_path, model_state_dict['opt'].config_path)
    config = get_config(config_path)
    config["DEFAULT"]["vocab_path"] = os.path.join(base_path, config["DEFAULT"]["vocab_path"])
    preprocessor = Preprocessor(input_type=InferInputType.INPUT_RAW_TXT, config=config)
    
    return {"model": model, "preprocessor": preprocessor}


def transform_fn(net, data, input_content_type, output_content_type):
    parse_data = __convert_data(data, input_content_type)
    print("[transform_fn] load data is {}".format(parse_data))
    print("[transform_fn] input_content_type is {}".format(input_content_type))
    print("[transform_fn] output_content_type is {}".format(output_content_type))
    
    results = inference(net, data=parse_data, num_of_summaries=1)
    print("[transform_fn] results is {}".format(results))
    
    # need to convert numpy array from list
    return __convert_result(results, output_content_type), output_content_type


def inference(net, data:str, num_of_summaries:int):
    model = net["model"]
    preprocessor = net["preprocessor"]
    executor = ModelExecutor(model=model)
    parse_data = preprocessor(data)
    return executor(data=parse_data, num_of_summaries=num_of_summaries)


def __convert_data(data, input_content_type):
    if input_content_type == "application/json":
        parse = json.loads(data)
        return parse
    elif input_content_type == "application/x-npy":
        return str(np.load(BytesIO(data)))
    else:
        raise NotImplementedError("{} is not supported!".format(input_content_type))

def __convert_result(result, output_content_type):
    if output_content_type == "application/json":
        result = {"result": result[0]}
        return json.dumps(result)
    elif output_content_type == "application/x-npy":
        raise NotImplementedError("numpy format is not supported!")
    else:
        raise NotImplementedError("{} format is not supported!".format(output_content_type))


def get_model_path(dir_path):
    file_list = glob.glob(os.path.join(dir_path, "model_step_*.pt"))
    model_path = file_list[0]
    return model_path


def get_config(config_path):
    config = ConfigParser()
    config.read(config_path, encoding='utf-8')
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--train_from", default='', type=str)
    parser.add_argument("--pretrained_model_path", default='./model/Japanese/', type=str)

    parser.add_argument("--train_steps", default=1200000, type=int)
    parser.add_argument("--report_every", default=10, type=int)
    parser.add_argument("--save_checkpoint_steps", default=1000, type=int)
    parser.add_argument("--accum_count", default=2, type=int)

    parser.add_argument("--optim", default='adam', type=str)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--decay_method", default='no', type=str)
    parser.add_argument("--warmup_steps", default=10000, type=int)

    parser.add_argument("--ff_size", default=2048, type=int)
    parser.add_argument("--heads", default=8, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--param_init", default=0.0, type=float)
    parser.add_argument("--param_init_glorot", default=True, type=bool)
    parser.add_argument("--max_grad_norm", default=0, type=float)
    parser.add_argument("--inter_layers", default=2, type=int)

    parser.add_argument("--seed", default='')
    parser.add_argument("--device", type=str, default="cuda", help="set the device name to calculate model parameter. 'cuda' or 'cpu'")
    parser.add_argument("--config_path", type=str, default="config.ini")
    
    # set the parameters for sagemaker
    parser.add_argument('--save_path', type=str, default=os.environ['SM_MODEL_DIR'] if 'SM_MODEL_DIR' in os.environ else '')
    parser.add_argument('--data_folder', type=str, default=os.environ['SM_CHANNEL_TRAIN'] if 'SM_CHANNEL_TRAIN' in os.environ else '')
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'] if 'SM_CURRENT_HOST' in os.environ else '')
    if 'SM_HOSTS' in os.environ:
        parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'] if 'SM_NUM_GPUS' in os.environ else 1)
    
    args = parser.parse_args()

    model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']

    device = args.device
    device_id = -1

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    config = get_config(args.config_path)
    
    def train_loader_fct():
        return DataLoader(args.data_folder, 512, args.batch_size, config=config, device=device, shuffle=True)

    model_params = SummarizerParams.from_args(args=args)
    model        = Summarizer(params=model_params)
    if args.train_from != '':
        print('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = dict(checkpoint['opt'])
        for k in opt.keys():
            if k in model_flags:
                setattr(args, k, opt[k])
        model.load_cp(checkpoint['model'])
        optim = build_optim(args, model, checkpoint)
    else:
        optim = build_optim(args, model, None)

    trainer = build_trainer(args, model, optim)
    trainer.train(train_loader_fct, args.train_steps)

