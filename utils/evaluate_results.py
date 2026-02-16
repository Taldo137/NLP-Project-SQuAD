import csv
import json
import evaluate


NO_ANSWER_MARKER = 'NO ANSWER'


def evaluate_results(datafile, final_answer_column='final_answer'):
    """
    Evaluate model predictions stored in a CSV file using the SQuAD v2.0 metric.

    This function reads a CSV containing ground-truth answers and model outputs,
    converts each row into the input format expected by Hugging Face's
    `evaluate` implementation of the `squad_v2` metric, and returns the computed
    evaluation results.

    The metric is designed for extractive QA with unanswerable questions (SQuAD v2.0),
    and reports Exact Match (EM) and token-level F1, along with breakdowns for
    answerable vs. unanswerable examples.
    """

    metric = evaluate.load("squad_v2")

    predictions = list()
    references =  list()

    with open(datafile, 'r', newline="", encoding="utf-8") as fin:
        csv_reader = csv.reader(fin)
        header = csv_reader.__next__()  # skip the header

        gt_answer_ind = header.index('answers')
        is_impossible_ind = header.index('is_impossible')
        final_model_answer_ind = header.index(final_answer_column)

        for i, line in enumerate(csv_reader):
            gt_answer = json.loads(line[gt_answer_ind])
            is_impossible = (line[is_impossible_ind].lower == "true")
            model_answer = line[final_model_answer_ind]

            predictions.append(
                {"id": str(i),
                 "prediction_text": (model_answer if model_answer != NO_ANSWER_MARKER else ''),
                 "no_answer_probability": (0.0 if model_answer != NO_ANSWER_MARKER else 1.0)}
            )

            references.append(
                {"id": str(i),
                 "answers": {
                     "text": ([] if is_impossible else [answer['text'] for answer in gt_answer]),
                     "answer_start": ([] if is_impossible else [0])
                    }
                 }
            )

    results = metric.compute(predictions=predictions, references=references)
    return results
