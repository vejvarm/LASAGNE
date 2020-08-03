import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='ConvQA')

    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--cuda_device', default=0, type=int)

    # data
    parser.add_argument('--data_path', default='/data/final/csqa/sample3')

    # experiments
    parser.add_argument('--snapshots', default='experiments/snapshots', type=str)
    parser.add_argument('--path_results', default='experiments/results', type=str)
    parser.add_argument('--path_error_analysis', default='experiments/error_analysis', type=str)
    parser.add_argument('--path_inference', default='experiments/inference', type=str)

    # task
    parser.add_argument('--task', default='multi_task', choices=['multi_task', 'ner', 'coref', 'logical_form'], type=str)

    # model
    parser.add_argument('--embDim', default=300, type=int)
    parser.add_argument('--dropout', default=0.1, type=int)
    parser.add_argument('--heads', default=6, type=int)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--max_positions', default=500, type=int)
    parser.add_argument('--pf_dim', default=300, type=int)

    # training
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--warmup', default=4000, type=float)
    parser.add_argument('--factor', default=1, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=1, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--clip', default=5, type=int)
    parser.add_argument('--batch_size', default=100, type=int)

    # test and inference
    parser.add_argument('--model_path', default='experiments/snapshots/ConvQA_model_e12_v-0.0057.pth.tar', type=str)
    parser.add_argument('--inference_partition', default='test', choices=['val', 'test'], type=str)
    # question type
    parser.add_argument('--question_type', default='Simple Question (Direct)',
        choices=['Clarification',
                'Comparative Reasoning (All)',
                'Logical Reasoning (All)',
                'Quantitative Reasoning (All)',
                'Simple Question (Coreferenced)',
                'Simple Question (Direct)',
                'Simple Question (Ellipsis)',
                'Verification (Boolean) (All)',
                'Quantitative Reasoning (Count) (All)',
                'Comparative Reasoning (Count) (All)'], type=str)

    return parser
