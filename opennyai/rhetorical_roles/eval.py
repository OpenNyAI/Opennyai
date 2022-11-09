import torch
from tqdm import tqdm
from wasabi import msg

from .utils import tensor_dict_to_gpu, tensor_dict_to_cpu


def infer_model(model, eval_batches, device, task, verbose=False):
    model.eval()
    labels_dict = {}
    predicted_labels = []
    doc_name_list = []
    docwise_predicted_labels = []
    if verbose:
        msg.info('Processing documents with rhetorical role model!!!')
    with torch.no_grad():
        for batch in tqdm(eval_batches, disable=not verbose):
            # move tensor to gpu
            tensor_dict_to_gpu(batch, device)

            if batch["task"] != task.task_name:
                continue

            output = model(batch=batch)

            true_labels_batch, predicted_labels_batch = \
                clear_and_map_padded_values(batch["label_ids"].view(-1), output["predicted_label"].view(-1),
                                            task.labels)
            doc_name_list.append(batch['doc_name'][0])
            docwise_predicted_labels.append(predicted_labels_batch)

            predicted_labels.extend(predicted_labels_batch)
            tensor_dict_to_cpu(batch)

    labels_dict['y_predicted'] = predicted_labels
    labels_dict['labels'] = task.labels
    labels_dict['doc_names'] = doc_name_list
    labels_dict['docwise_y_predicted'] = docwise_predicted_labels

    return labels_dict


def clear_and_map_padded_values(true_labels, predicted_labels, labels):
    assert len(true_labels) == len(predicted_labels)
    cleared_predicted = []
    cleared_true = []
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        # filter masked labels (0)
        if true_label > 0:
            cleared_true.append(labels[true_label])
            cleared_predicted.append(labels[predicted_label])
    return cleared_true, cleared_predicted
