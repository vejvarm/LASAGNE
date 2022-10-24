import argparse

question_types = ['Clarification',
                  'Comparative Reasoning (All)',
                  'Logical Reasoning (All)',
                  'Quantitative Reasoning (All)',
                  'Simple Question (Coreferenced)',
                  'Simple Question (Direct)',
                  'Simple Question (Ellipsis)',
                  'Verification (Boolean) (All)',
                  'Quantitative Reasoning (Count) (All)',
                  'Comparative Reasoning (Count) (All)']


def get_parser():
    parser = argparse.ArgumentParser(description='LASAGNE')

    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--cuda_device', default=0, type=int)

    # data
    parser.add_argument('--data_path', default='/home/freya/PycharmProjects/CARTON/data/final/csqa/')
    parser.add_argument('--kg_path', default='/home/freya/PycharmProjects/CARTON/knowledge_graph/Wikidata.fs')

    # experiments
    parser.add_argument('--snapshots', default='experiments/snapshots', type=str)
    parser.add_argument('--path_results', default='experiments/results', type=str)
    parser.add_argument('--path_error_analysis', default='experiments/error_analysis', type=str)
    parser.add_argument('--path_inference', default='experiments/inference', type=str)

    # task
    parser.add_argument('--task', default='multitask', choices=['multitask',
                                                                'logical_form',
                                                                'ner',
                                                                'coref',
                                                                'graph'], type=str)

    # model
    parser.add_argument('--emb_dim', default=300, type=int)
    parser.add_argument('--dropout', default=0.1, type=int)
    parser.add_argument('--heads', default=6, type=int)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--max_positions', default=1000, type=int)
    parser.add_argument('--pf_dim', default=600, type=int)
    parser.add_argument('--graph_heads', default=2, type=int)
    parser.add_argument('--bert_dim', default=3072, type=int)

    # training
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--warmup', default=4000, type=float)
    parser.add_argument('--factor', default=1, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=1, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--clip', default=5, type=int)
    parser.add_argument('--batch_size', default=50, type=int)

    # test and inference
    parser.add_argument('--model_path', default='experiments/models/LASAGNE_e19_v0.0167_multitask.pth.tar', type=str)
    parser.add_argument('--file_path', default='/experiments/inference/LASAGNE_e19_v0.0167_multitask_test_Clarification.json', type=str)
    parser.add_argument('--inference_partition', default='test', choices=['val', 'test'], type=str)
    parser.add_argument('--question_type', default='Clarification', choices=question_types, type=str)
    # parser.add_argument('--question_types_to_run', default=['Clarification']*10)  # ANCHOR: For testing purposes
    parser.add_argument('--question_types_to_run', default=question_types)

    # elasticsearch related
    parser.add_argument('--elastic_host', default='https://localhost:9200')
    parser.add_argument('--elastic_certs', default='./knowledge_graph/certs/http_ca.crt')
    parser.add_argument('--elastic_user', default='elastic')
    parser.add_argument('--elastic_password', default='1jceIiR5k6JlmSyDpNwK')  # Notebook: hZiYNU+ye9izCApoff-v

    return parser