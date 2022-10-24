import random
import logging
import torch
import numpy as np
from model import LASAGNE
from dataset import CSQADataset
from utils import Predictor, Inference

import threading

# import constants
from constants import *

# set logger
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(f'{args.path_results}/inference_{args.question_type}.log', 'w'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# set a seed value
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main():
    # load data
    dataset = CSQADataset()
    vocabs = dataset.get_vocabs()
    inference_data = dataset.get_inference_data(args.inference_partition)

    logger.info(f'Inference partition: {args.inference_partition}')
    logger.info('Inference data prepared')
    logger.info(f"Num of inference data: {len(inference_data)}")

    # load model
    model = LASAGNE(vocabs).to(DEVICE)

    logger.info(f"=> loading checkpoint '{args.model_path}'")
    if DEVICE.type=='cpu':
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1')
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"=> loaded checkpoint '{args.model_path}' (epoch {checkpoint['epoch']})")

    # construct actions
    predictor = Predictor(model, vocabs, DEVICE)
    inference = Inference()
    threads = []
    for i, question_type in enumerate(args.question_types_to_run):
        logger.info(f'Inference question type: {question_type}')
        threads.append(threading.Thread(target=inference.construct_actions, args=(inference_data, predictor, question_type)))
    # Inference().construct_actions(inference_data, predictor, args.question_type)

    for t in threads:
        t.start()

    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
