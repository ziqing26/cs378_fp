import numpy as np
import collections
from collections import defaultdict, OrderedDict
from transformers import Trainer, EvalPrediction
from transformers.trainer_utils import PredictionOutput
from typing import Tuple
from tqdm.auto import tqdm
from util import Indexer
from collections import defaultdict
import string
import matplotlib.pyplot as plt
import json

QA_MAX_ANSWER_LENGTH = 30


# This function preprocesses an NLI dataset, tokenizing premises and hypotheses.
def prepare_dataset_nli(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    tokenized_examples = tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )

    tokenized_examples['label'] = examples['label']
    return tokenized_examples


# This function computes sentence-classification accuracy.
# Functions with signatures like this one work as the "compute_metrics" argument of transformers.Trainer.
def compute_accuracy(eval_preds: EvalPrediction):
    return {
        'accuracy': (np.argmax(
            eval_preds.predictions,
            axis=1) == eval_preds.label_ids).astype(
            np.float32).mean().item()
    }


    # print("train_dataset length", len(dataset))
    # print("predict length", len(eval_preds.label_ids))
    # {
    #     index of word: (num of class 0 prediction, num of class 1 pred, num of class 2 pred)
    # }
    # indexer = Indexer()
    count = defaultdict(lambda: [0, 0, 0])

    for i in range(len(eval_preds.label_ids)):
        pred = eval_preds.predictions[i].argmax()
        sentence = dataset['premise'][i]
        sentence = ''.join(list(map(str.lower, sentence)))
        sentence = sentence.translate(
            str.maketrans('', '', string.punctuation))
        sentence = sentence.split(' ')

        for word in sentence:
            # index = indexer.add_and_get_index(word)
            count[word][pred] += 1

    y0, y1, y2, x = [], [], [], []

    for word, counts in list(count.items()):
        counts = np.array(counts)
        total = np.sum(counts)
        probs = counts / total
        x.append(total)
        y0.append(probs[0])
        y1.append(probs[1])
        y2.append(probs[2])

    n = np.arange(1, max(x))
    z = 0.95
    p = ((2/(9*n))**0.5)*z + 1/3
    plt.plot(n, p, label=r'$\alpha = 0.05$')

    plot_graph(x, y0, "blue", "entailment")
    plot_graph(x, y1, "red", "neutral")
    plot_graph(x, y2, "green", "contradiction")

    plt.xscale('log', base=10)
    plt.xlim(min(x), max(x))
    plt.legend(loc="upper right")
    plt.savefig("aggregate"+".png")
    return
def compute_graph(eval_preds: EvalPrediction, dataset):
    # print("train_dataset length", len(dataset))
    # print("predict length", len(eval_preds.label_ids))
    # {
    #     index of word: (num of class 0 prediction, num of class 1 pred, num of class 2 pred)
    # }
    count = defaultdict(lambda: [0, 0, 0])

    sentences = [] #(sentence, id, pred_label_id)
    for i in range(len(eval_preds.label_ids)):
        if i % 500 == 0:
            print("number of sentence:", i)
        pred = eval_preds.predictions[i].argmax()
        sentence = dataset['hypothesis'][i]
        sentence = ''.join(list(map(str.lower, sentence)))
        sentence = sentence.translate(
            str.maketrans('', '', string.punctuation))
        sentence = sentence.split(' ')
        
        sentences.append((sentence, i, pred))

        for word in sentence:
            # index = indexer.add_and_get_index(word)
            count[word][pred] += 1

    y0, y1, y2, x = [], [], [], []

    index = 0
    for word, counts in list(count.items()):
        if index % 1000 == 0:
            print("number of word:", index)
        index += 1
        counts = np.array(counts)
        total = np.sum(counts)
        probs = counts / total
        # compare probs with p
        #   (word1, probs, label), word2, ...]
        # find the sentence
        x.append(total)
        y0.append(probs[0])
        y1.append(probs[1])
        y2.append(probs[2])

    n = np.arange(1, max(x))
    z = 0.95
    p = ((2/(9*n))**0.5)*z + 1/3
    ############# Local Edits #############
    # print("===============LOCAL EDIT START====================")
    # local_edits_list = []
    # for idx, word in enumerate(list(count.keys())):
    #     n_i = x[idx]
    #     p_i = ((2/(9*n_i))**0.5)*z + 1/3
    #     if sum(list(count[word])) < 3:
    #         continue
    #     if y0[idx] > p_i:
    #         local_edits_list.append((word, y0[idx], 0))
    #     if y1[idx] > p_i:
    #         local_edits_list.append((word, y1[idx], 1))
    #     if y2[idx] > p_i:
    #         local_edits_list.append((word, y2[idx], 2))

    # # sort by the distance from the hypothesis test
    # local_edits_list.sort(key=lambda x: x[1], reverse=True)

    # num_to_edit = 20
    # if len(local_edits_list) < 20:
    #     num_to_edit = len(local_edits_list)
    # print ("local_edits_list", len(local_edits_list))

    # # get id of the sentences that needs to be edited
    # res_dict = defaultdict(lambda : [])
    # res_dict_len = 0
    # with open('local_edit_target.json', encoding='utf-8', mode='w') as f:
    #     for idx, sentence in enumerate(sentences):
    #         texts, sidx, pred_label = sentence
    #         for word, prob, label in local_edits_list[:num_to_edit]:
    #             if pred_label == label and word in texts:
    #                 # print(dataset.select([sidx]), file = f)
    #                 distrib = ','.join([str(x) for x in count[word]])
    #                 name = word+"("+ distrib +")"
    #                 res_dict[name].append(dataset.select([sidx])[0])
    #                 res_dict_len += 1
    #                 break

    #     print("res_dict_len", res_dict_len)
    #     json.dump(res_dict, f)

    # print("===============LOCAL EDIT END====================")
    ############# Local Edits End #############
    plt.plot(n, p, label=r'$\alpha = 0.05$')

    plot_graph(x, y0, "blue", "entailment")
    plot_graph(x, y1, "red", "neutral")
    plot_graph(x, y2, "green", "contradiction")

    plt.xscale('log', base=10)
    plt.xlim(min(x), max(x))
    plt.legend(loc="upper right")
    plt.savefig("aggregate"+".png")
    return

def plot_graph(x, y, color, name):
    plt.scatter(x, y, c=color, s=1, label=name)
    plt.xlabel("n")
    plt.ylabel(r'$\hat p (y|x_i)$')

# This function preprocesses a question answering dataset, tokenizing the question and context text
# and finding the right offsets for the answer spans in the tokenized context (to use as labels).
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py


def prepare_train_dataset_qa(examples, tokenizer, max_seq_length=None):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    # tokenize both questions and the corresponding context
    # if the context length is longer than max_length, we split it to several
    # chunks of max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        # padding="max_length"
        padding=True
    )

    # Since one example might give us several features if it has a long context,
    # we need a map from a feature to its corresponding example.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position
    # in the original context. This will help us compute the start_positions
    # and end_positions to get the final answer string.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        # We will label features not containing the answer the index of the CLS token.
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        # from the feature idx to sample idx
        sample_index = sample_mapping[i]
        # get the answer for a feature
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and \
                        offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(
                    token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_dataset_qa(examples, tokenizer):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


# This function uses start and end position scores predicted by a question answering model to
# select and extract the predicted answer span from the context.
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/utils_qa.py
def postprocess_qa_predictions(examples,
                               features,
                               predictions: Tuple[np.ndarray, np.ndarray],
                               n_best_size: int = 20):
    if len(predictions) != 2:
        raise ValueError(
            "`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(
            f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[
            example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits
            # to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                -1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[
                -1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or \
                            end_index - start_index + 1 > QA_MAX_ANSWER_LENGTH:
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0],
                                        offset_mapping[end_index][1]),
                            "score": start_logits[start_index] +
                            end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"],
                             reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]: offsets[1]]

        # In the very rare edge case we have not a single non-null prediction,
        # we create a fake prediction to avoid failure.
        if len(predictions) == 0 or (
                len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0,
                                   "end_logit": 0.0, "score": 0.0})

        all_predictions[example["id"]] = predictions[0]["text"]
    return all_predictions


# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples

    def evaluate(self,
                 eval_dataset=None,  # denotes the dataset after mapping
                 eval_examples=None,  # denotes the raw dataset
                 ignore_keys=None,  # keys to be ignored in dataset
                 metric_key_prefix: str = "eval"
                 ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            # compute the raw predictions (start_logits and end_logits)
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            # post process the raw predictions to get the final prediction
            # (from start_logits, end_logits to an answer string)
            eval_preds = postprocess_qa_predictions(eval_examples,
                                                    eval_dataset,
                                                    output.predictions)
            formatted_predictions = [{"id": k, "prediction_text": v}
                                     for k, v in eval_preds.items()]
            references = [{"id": ex["id"], "answers": ex['answers']}
                          for ex in eval_examples]

            # compute the metrics according to the predictions and references
            metrics = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions,
                               label_ids=references)
            )

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state,
                                                         self.control, metrics)
        return metrics
