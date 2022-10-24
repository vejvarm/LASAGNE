import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import threading
import time
import json
import argparse
from glob import glob
from pathlib import Path
import ZODB
from knowledge_graph.ZODBConnector import BTreeDBConn
from knowledge_graph.knowledge_graph import KnowledgeGraph
from args import get_parser
from executor import ActionExecutor
from meters import AccuracyMeter, F1scoreMeter
ROOT_PATH = Path(os.path.dirname(__file__)).parent


# add arguments to parser
parser = get_parser()
parser.add_argument('--max_results', default=1000, help='json file with actions')
args = parser.parse_args()

# load kg
# kg = KnowledgeGraph()
db = ZODB.DB(args.kg_path, pool_size=len(args.question_types_to_run))

# define question type meters
question_types_meters = {
    'Clarification': F1scoreMeter(),
    'Comparative Reasoning (All)': F1scoreMeter(),
    'Logical Reasoning (All)': F1scoreMeter(),
    'Quantitative Reasoning (All)': F1scoreMeter(),
    'Simple Question (Coreferenced)': F1scoreMeter(),
    'Simple Question (Direct)': F1scoreMeter(),
    'Simple Question (Ellipsis)': F1scoreMeter(),
    # -------------------------------------------
    'Verification (Boolean) (All)': AccuracyMeter(),
    'Quantitative Reasoning (Count) (All)': AccuracyMeter(),
    'Comparative Reasoning (Count) (All)': AccuracyMeter()
}


def run_question(file_path=args.file_path, question_type=args.question_type):
    kg = BTreeDBConn(db, run_adapter=True, multithread=True)
    # load action executor
    action_executor = ActionExecutor(kg)

    # load data
    data_path = f'{str(ROOT_PATH)}{file_path}'
    print(data_path)
    with open(data_path) as json_file:
        data = json.load(json_file)

    print(len(data))

    count_no_answer = 0
    count_total = 0
    tic = time.perf_counter()
    for i, d in enumerate(data):

        if question_type not in d['question_type']:
            continue
        else:
            if question_type == 'Clarification':
                current_question = d['question_type'][0]  # 'Clarification'
            else:
                current_question = d['question_type'][-1]

        print(current_question)

        count_total += 1
        try:
            if d['actions'] is not None:
                all_actions = [action[1] for action in d['actions']]
                if 'entity' not in all_actions:
                    result = action_executor(d['actions'], d['prev_results'], current_question)
                else:
                    count_no_answer += 1
                    result = set([])
            else:
                count_no_answer += 1
                result = set([])
        except Exception as ex:
            print(d['question'])
            print(d['actions'])
            print(ex)
            count_no_answer += 1
            result = set([])

        try:
            if current_question == 'Verification (Boolean) (All)':
                answer = True if d['answer'] == 'YES' else False
                question_types_meters[current_question].update(answer, result)
            else:
                if current_question in ['Quantitative Reasoning (Count) (All)', 'Comparative Reasoning (Count) (All)']:
                    if d['answer'].isnumeric():
                        question_types_meters[current_question].update(int(d['answer']), len(result))
                    else:
                        question_types_meters[current_question].update(len(d['results']), len(result))
                else:
                    if result != set(d['results']) and len(result) > args.max_results:
                        new_result = result.intersection(set(d['results']))
                        for res in result:
                            if res not in result: new_result.add(res)
                            if len(new_result) == args.max_results: break
                        result = new_result.copy()
                    gold = set(d['results'])
                    question_types_meters[current_question].update(gold, result)
        except Exception as ex:
            print(d['question'])
            print(d['actions'])
            raise ValueError(ex)

        toc = time.perf_counter()
        print(f'==> Finished {((i+1)/len(data))*100:.2f}% -- {toc - tic:0.2f}s')

    # print results
    print(question_type)
    print(f'NA actions: {count_no_answer}')
    print(f'Total samples: {count_total}')
    if question_type in ['Verification (Boolean) (All)', 'Quantitative Reasoning (Count) (All)', 'Comparative Reasoning (Count) (All)']:
        print(f'Accuracy: {question_types_meters[question_type].accuracy}')
    else:
        print(f'Precision: {question_types_meters[question_type].precision}')
        print(f'Recall: {question_types_meters[question_type].recall}')
        print(f'F1-score: {question_types_meters[question_type].f1_score}')


if __name__ == '__main__':
    file_path_root = '/experiments/inference/LASAGNE_e19_v0.0167_multitask_test_'
    question_types = args.question_types_to_run

    threads = []
    file_paths = []

    for qt in question_types:
        file_paths.append(file_path_root+qt+'.json')
        threads.append(threading.Thread(target=run_question, args=(file_paths[-1], qt)))

    for t in threads:
        t.start()

    for t in threads:
        t.join()
