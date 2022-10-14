import sys
import torch
import datetime
from threading import Lock

from prettytable import PrettyTable


def log(str):
    print(str, file=sys.stderr)
    sys.stderr.flush()


def get_device(gpu_device=0):
    device = torch.device(f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu")
    return device


def tensor_dict_to_gpu(d, device):
    tensor_dict_to_device(d, device)


def tensor_dict_to_cpu(d):
    tensor_dict_to_device(d, torch.device("cpu"))


def tensor_dict_to_device(d, device):
    for k, v in d.items():
        if torch.is_tensor(v):
            d[k] = v.to(device)
        else:
            d[k] = v



class ResultWriter:
    def __init__(self, results_filename):
        self.results_filename = results_filename
        self.lock = Lock()

    def write(self, str):
        self.lock.acquire()
        try:
            with open(self.results_filename, "a", encoding="utf-8") as f:
                f.write(str + "\n")
        finally:
            self.lock.release()

    def log(self, msg):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        msg = timestamp + ": " + msg
        log(msg)
        self.lock.acquire()
        try:
            with open(self.results_filename + ".log", "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        finally:
            self.lock.release()


def get_num_model_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def print_model_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, f'{param:,}'])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

