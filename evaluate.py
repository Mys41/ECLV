from collections import Counter
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from nltk import ngrams
from nltk.translate import bleu_score
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from fairseq import utils
import sys
sys.path.append('/home/taoran/ECLV')
from src.task import DialogVEDTask
import math

import argparse


# references:
# https://github.com/lemuria-wchen/Research/blob/master/NLP/Dialogue-PLATO/plato/metrics/metrics.py
# https://github.com/lemuria-wchen/Research/blob/master/NLP/Dialogue-PLATO/tools/dstc7_avsd_eval.py
# https://github.com/lemuria-wchen/Research/blob/master/NLP/Dialogue-PLATO/tools/knowledge_f1.py

# This script integrates all evaluation methods proposed in the Plato article.
# ACL 2020: https://www.aclweb.org/anthology/2020.acl-main.9.pdf
# Repository: https://github.com/PaddlePaddle/Research/tree/master/NLP/Dialogue-PLATO


def detokenize_bert(string: str) -> str:
    return string.replace(' ##', '')


# hyps and refs data loader for dailydialog & personachat
def load_file(hyps_file_path: str, refs_file_path=None, ignore_indices=None) -> tuple:
    with open(hyps_file_path, 'r', encoding='utf-8') as f:
        hyps_fin = f.readlines()
        hyps = [line.strip().split() for line in hyps_fin]
    refs = None
    if refs_file_path:
        with open(refs_file_path, 'r', encoding='utf-8') as f:
            refs_fin = f.readlines()
            if ignore_indices:
                assert isinstance(ignore_indices, list)
                refs_fin = [line for idx, line in enumerate(refs_fin) if idx not in ignore_indices]
        refs = [detokenize_bert(line.strip()).split() for line in refs_fin]
    return hyps, refs


# hyps and refs data loader for dstc7avsd
def _load_file(hyps_file_path: str, refs_file_path=None) -> tuple:
    # load predicted file and reference file
    with open(hyps_file_path, 'r', encoding='utf-8') as f:
        hyps_fin = f.readlines()
        hyps = [line.strip() for line in hyps_fin]
    with open(refs_file_path, 'r', encoding='utf-8') as f:
        refs, tmp = [], []
        for line in f.readlines():
            if line != '\n':
                tmp.append(detokenize_bert(line.strip()))
            else:
                refs.append(tmp)
                tmp = []
    assert len(hyps) == len(refs), 'number of instances of hyps and refs muse be equal'
    return hyps, refs


# calculate BLEU-1/2/3/4 for dailydialog & personachat & esconv
def bleu(hyps_file_path: str, refs_file_path: str, ignore_indices=None, output_path=None) -> tuple:
    hyp_list = []
    ref_list = []
    f1 = open(hyps_file_path, mode='r', encoding='utf-8')
    f2 = open(refs_file_path, mode='r', encoding='utf-8')
    for line in f1.readlines():
        hyp_list.append(line.strip('[unused0]').strip('[unused1]'))
    for line in f2.readlines():
        ref_list.append(line)
    ref_list = ref_list[:len(hyp_list)]
    hypothesis = [hyp.split() for hyp in hyp_list]
    references = [[ref.split()] for ref in ref_list]
    b1 = corpus_bleu(references, hypothesis, weights=(1.0 / 1.0,),
                     smoothing_function=bleu_score.SmoothingFunction(epsilon=1e-12).method1)
    b2 = corpus_bleu(references, hypothesis, weights=(1.0 / 2.0, 1.0 / 2.0),
                     smoothing_function=bleu_score.SmoothingFunction(epsilon=1e-12).method1)
    b3 = corpus_bleu(references, hypothesis, weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
                     smoothing_function=bleu_score.SmoothingFunction(epsilon=1e-12).method1)
    b4 = corpus_bleu(references, hypothesis, weights=(1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0),
                     smoothing_function=bleu_score.SmoothingFunction(epsilon=1e-12).method1)
    output_content = 'BLEU-1/2/3/4: {}/{}/{}/{}\n'.format(round(b1*100, 2), round(b2*100, 2)
                                                          , round(b3*100, 2), round(b4*100, 2))
    print('-------------- BLEU score --------------\n{}'.format(output_content))
    if output_path is not None:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(output_content)
    return b1, b2, b3, b4


# calculate Distinct-1/2 for esconv
def distinct(hyps_file_path: str, output_path=None) -> tuple:
    hyp_list = []
    f1 = open(hyps_file_path, mode='r', encoding='utf-8')
    for line in f1.readlines():
        hyp_list.append(line.strip('[unused0]').strip('[unused1]'))
    unigram_counter, bigram_counter = Counter(), Counter()
    for hypo in hyp_list:
        tokens = hypo.split()
        unigram_counter.update(tokens)
        bigram_counter.update(ngrams(tokens, 2))

    distinct_1 = len(unigram_counter) / sum(unigram_counter.values())
    distinct_2 = len(bigram_counter) / sum(bigram_counter.values())

    output_content = 'Distinct-1/2: {}/{}\n'.format(round(distinct_1*100, 2), round(distinct_2*100, 2))
    print('-------------- Distinct score --------------\n{}'.format(output_content))
    if output_path is not None:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(output_content)
    return distinct_1, distinct_2


# calculate Rouge-L for esconv
def rouge(hyps_file_path: str, refs_file_path: str, ignore_indices=None, output_path=None) -> tuple:
    # 初始化Rouge对象
    rouge = Rouge()
    # 从文件中读取候选句子和参考句子
    with open(hyps_file_path, 'r') as f:
        hyps = {str(i): [line.strip()] for i, line in enumerate(f)}
    with open(refs_file_path, 'r') as f:
        refs = {str(i): [line.strip()] for i, line in enumerate(f)}

    # 计算ROUGE-L得分
    score_map = rouge.compute_score(refs, hyps)

    output_content = 'Rouge-L: {}\n'.format(round(score_map[0]*100, 2))
    print('-------------- Rouge-L score --------------\n{}'.format(output_content))
    if output_path is not None:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(output_content)
    return score_map[0]


# calculate ACC% for esconv
def acc(hyps_file_path: str, refs_file_path: str, output_path=None) -> float:
    # 从文件中读取候选句子和参考句子
    hyp_list = []
    ref_list = []
    f1 = open(hyps_file_path, mode='r', encoding='utf-8')
    f2 = open(refs_file_path, mode='r', encoding='utf-8')
    for line in f1.readlines():
        hyp_list.append(line.strip('[unused0]').strip('[unused1]'))
    for line in f2.readlines():
        ref_list.append(line)
    ref_list = ref_list[:len(hyp_list)]
    hypothesis = [hyp.split() for hyp in hyp_list]
    references = [ref.split() for ref in ref_list]

    # 计算策略预测准确率
    assert len(hypothesis) == len(references), "The two lists must have the same length."
    total = len(hypothesis)
    correct = 0
    for h, r in zip(hypothesis, references):
        if h[0] == r[0]:
            correct += 1
    acc = correct / total

    output_content = 'ACC: {}\n'.format(round(acc * 100, 2))
    print('-------------- ACC score --------------\n{}'.format(output_content))
    if output_path is not None:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(output_content)
    return acc


# calculate ppl for esconv
# def ppl(hyps_file_path: str, output_path=None) -> float:
#     # register_task('ved_translate')(DialogVEDTask)
#     models, cfg, task = load_model_ensemble_and_task(
#         ['/nfs/home/taoran/DialogVED-main/data/finetune/esconv/checkpoints/checkpoint_best.pt']
#     )
#     model = models[0]
#     task.load_dataset('test', data_path='/nfs/home/taoran/DialogVED-main/data/finetune/esconv/binary')
#     itr = task.get_batch_iterator(
#         dataset=task.dataset('test'),
#         max_tokens=512,
#         max_sentences=32,
#         max_positions=utils.resolve_max_positions(
#             task.max_positions(),
#             model.max_positions()
#         ),
#         ignore_invalid_inputs=True,
#         num_shards=1,
#         shard_id=0,
#     ).next_epoch_itr(shuffle=False)
#
#     log_sum = 0
#     count = 0
#
#     for sample in itr:
#         sample = utils.move_to_cuda(sample)
#         net_output = model(**sample['net_input'])
#         lprobs = model.get_normalized_probs(net_output[0][0], log_probs=True)
#         target = model.get_targets(sample, net_output)
#         log_prob = lprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
#         log_sum += log_prob.sum().item()
#         count += log_prob.numel()
#
#     perplexity = math.exp(-log_sum / count)
#
#     output_content = 'PPL: {}\n'.format(round(perplexity, 4))
#     print('-------------- PPL score --------------\n{}'.format(output_content))
#     if output_path is not None:
#         with open(output_path, 'a', encoding='utf-8') as f:
#             f.write(output_content)
#     return perplexity


# calculate knowledge f1 for personachat
def knowledge_f1(hyps_file_path: str, refs_file_path: str, output_path=None) -> tuple:
    # load stopwords
    stopwords = set()
    with open('./stopwords.txt', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            stopwords.add(word)
    # load predicted file and reference file
    with open(hyps_file_path, 'r', encoding='utf-8') as f:
        hyps_fin = f.readlines()
    with open(refs_file_path, 'r', encoding='utf-8') as f:
        refs_fin = f.readlines()
    hyps = [line.strip() for line in hyps_fin]
    refs = [line.strip() for line in refs_fin]
    assert len(hyps) == len(refs), 'number of instances of hyps and refs muse be equal'
    # calculate knowledge f1 value
    cnt, res, r, p = 0, .0, .0, .0
    for hyp, ref in zip(hyps, refs):
        cnt += 1
        # prediction
        hyp = set(hyp.split())
        hyp = hyp - stopwords
        hyp_len = len(hyp)
        # reference
        knowledge, _, _ = ref.strip().split('\t')
        words = set()
        for sent in knowledge.split(" __eou__ "):
            for word in sent.split():
                words.add(word)
        words = words - stopwords
        k_len = len(words)
        overlap = len(words & hyp)
        if overlap == 0:
            continue
        recall = float(overlap) / k_len
        r += recall
        precision = float(overlap) / hyp_len
        p += precision
        res += 2 * recall * precision / (recall + precision)
    # recall/precision/f1
    output_content = 'Knowledge R/P/F1: {}/{}/{}\n'.format(round(r / cnt, 4), round(p / cnt, 4), round(res / cnt, 4))
    print('-------------- Knowledge score --------------\n{}'.format(output_content))
    if output_path is not None:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(output_content)
    return r / cnt, p / cnt, res / cnt


# calculate BLEU-1/2/3/4, METEOR, ROUGH-L and CIDEr for dstc7avsd
def score_fn(_hyp: dict, _ref: dict):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    _scores = {}
    for scorer, method in scorers:
        _score, _ = scorer.compute_score(_ref, _hyp)
        if type(_score) == list:
            for m, s in zip(method, _score):
                _scores[m] = s
        else:
            _scores[method] = _score
    return _scores


# calculate BLEU-1/2/3/4, METEOR, ROUGH-L and CIDEr for dstc7avsd
def coco_eval(hyps_file_path: str, refs_file_path: str, output_path=None) -> dict:
    hyps, refs = _load_file(hyps_file_path, refs_file_path)
    hyps = {idx: [hyp] for idx, hyp in enumerate(hyps)}
    refs = {idx: ref for idx, ref in enumerate(refs)}
    res = score_fn(hyps, refs)
    output_content = ''
    for name in res:
        output_content += '{}: {}\n'.format(name, round(res[name], 4))
    print('-------------- Microsoft COCO score --------------\n{}'.format(output_content))
    if output_path is not None:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(output_content)
    return res


def evaluate(task_name: str, hyps_file_path: str, refs_file_path: str, output_path: str, knowledge_file_path=None):
    assert task_name in ['dailydialog', 'esconv', 'personachat', 'dstc7avsd'], \
        'now only the evaluation of the above tasks is supported.'

    if task_name == 'dailydialog':
        bleu(hyps_file_path, refs_file_path, output_path=output_path)
        distinct(hyps_file_path, output_path=output_path)
    elif task_name == 'esconv':
        bleu(hyps_file_path, refs_file_path, output_path=output_path)
        distinct(hyps_file_path, output_path=output_path)
        rouge(hyps_file_path, refs_file_path, output_path=output_path)
        acc(hyps_file_path, refs_file_path, output_path=output_path)
    elif task_name == 'personachat':
        bleu(hyps_file_path, refs_file_path, output_path=output_path)
        distinct(hyps_file_path, output_path=output_path)
        assert knowledge_file_path is not None, 'if evaluate personachat, knowledge file path must be provided'
        knowledge_f1(hyps_file_path, knowledge_file_path, output_path=output_path)
    else:
        coco_eval(hyps_file_path, refs_file_path, output_path=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='command line parameter for dialogue generation evaluation')

    parser.add_argument('-name', '--task_name', type=str, required=True, help='specify which task to evaluate')
    parser.add_argument('-hyp', '--hyps_file_path', type=str, required=True, help='predicted file path')
    parser.add_argument('-ref', '--refs_file_path', type=str, required=True, help='gold file path')
    parser.add_argument('-out', '--output_path', type=str, required=False, default=None, help='output path')
    parser.add_argument('-know', '--knowledge_file_path', type=str, required=False, default=None, help='knowledge path')
    args = parser.parse_args()

    evaluate(
        task_name=args.task_name, hyps_file_path=args.hyps_file_path,
        refs_file_path=args.refs_file_path, output_path=args.output_path,
        knowledge_file_path=args.knowledge_file_path,
    )
