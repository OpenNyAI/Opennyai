import os

from opennyai.utils.download import CACHE_DIR

EXTRACTIVE_SUMMARIZER_CACHE_PATH = os.path.join(CACHE_DIR, 'ExtractiveSummarizer'.lower())


class ARGS:
    def __init__(self):
        self.default = True


def __setargs__():
    parser = ARGS()
    parser.task = 'ext'
    parser.encoder = 'bert'
    parser.mode = 'test'
    parser.bert_data_path = EXTRACTIVE_SUMMARIZER_CACHE_PATH
    parser.model_path = EXTRACTIVE_SUMMARIZER_CACHE_PATH
    parser.result_path = EXTRACTIVE_SUMMARIZER_CACHE_PATH
    parser.temp_dir = EXTRACTIVE_SUMMARIZER_CACHE_PATH
    parser.batch_size = 5000
    parser.test_batch_size = 1
    parser.max_pos = 512
    parser.use_interval = True
    parser.large = False
    parser.load_from_extractive = ''
    parser.sep_optim = True
    parser.lr_bert = 2e-3
    parser.lr_dec = 2e-3
    parser.use_bert_emb = False
    parser.share_emb = False
    parser.finetune_bert = True
    parser.dec_dropout = 0.2
    parser.dec_layers = 6
    parser.dec_hidden_size = 768
    parser.dec_heads = 8
    parser.dec_ff_size = 2048
    parser.enc_hidden_size = 512
    parser.enc_ff_size = 512
    parser.enc_dropout = 0.2
    parser.enc_layers = 6
    # params for EXT
    parser.ext_dropout = 0.2
    parser.ext_layers = 2
    parser.ext_hidden_size = 768
    parser.ext_heads = 8
    parser.ext_ff_size = 2048
    parser.label_smoothing = 0.1
    parser.generator_shard_size = 32
    parser.alpha = 0.95
    parser.beam_size = 5
    parser.min_length = 0
    parser.max_length = 2000
    parser.max_tgt_len = 140
    parser.use_rhetorical_roles = True  #### whether to use rhetorical roles in the mode
    parser.seperate_summary_for_each_rr = False  #### whether to select top N sentences from each rhetorical role
    parser.rogue_exclude_roles_not_in_test = True  #### whether to remove the sections that are present in predicted summaries which are not in test data while ROGUE calculation
    parser.add_additional_mandatory_roles_to_summary = False  #### whether to add the additional mandatory roles to predicted summary
    parser.summary_sent_precent = 20  ##### top N percentage of sentences to be selected
    parser.use_adaptive_summary_sent_percent = True  ##### whether summary sentence percentage should be chosen as per input text sentence length
    parser.param_init = 0
    parser.param_init_glorot = True
    parser.optim = 'adam'
    parser.lr = 1
    parser.beta1 = 0.9
    parser.beta2 = 0.999
    parser.warmup_steps = 8000
    parser.warmup_steps_bert = 8000
    parser.warmup_steps_dec = 8000
    parser.max_grad_norm = 0
    parser.save_checkpoint_steps = 5
    parser.accum_count = 1
    parser.report_every = 1
    parser.train_steps = 1000
    parser.recall_eval = False
    parser.visible_gpus = '-1'
    parser.gpu_ranks = '0'
    parser.log_file = './log.log'
    parser.seed = 666
    parser.test_all = False
    parser.test_from = ''
    parser.test_start_from = -1
    parser.train_from = ''
    parser.report_rouge = True
    parser.block_trigram = True
    model_args = parser

    parser_preprocessing = ARGS()
    parser_preprocessing.pretrained_model = 'bert'

    parser_preprocessing.mode = 'format_to_bert'
    parser_preprocessing.select_mode = 'greedy'
    parser_preprocessing.map_path = EXTRACTIVE_SUMMARIZER_CACHE_PATH
    parser_preprocessing.raw_path = EXTRACTIVE_SUMMARIZER_CACHE_PATH
    parser_preprocessing.save_patt = EXTRACTIVE_SUMMARIZER_CACHE_PATH

    parser_preprocessing.shard_size = 2000
    parser_preprocessing.min_src_nsents = 0
    parser_preprocessing.max_src_nsents = 50000
    parser_preprocessing.min_src_ntokens_per_sent = 0
    parser_preprocessing.max_src_ntokens_per_sent = 512
    parser_preprocessing.min_tgt_ntokens = 0
    parser_preprocessing.max_tgt_ntokens = 20000

    parser_preprocessing.lower = True
    parser_preprocessing.use_bert_basic_tokenizer = False

    parser_preprocessing.log_file = os.path.join(EXTRACTIVE_SUMMARIZER_CACHE_PATH, 'log.log')

    parser_preprocessing.dataset = ''

    parser_preprocessing.n_cpus = 2
    return model_args, parser_preprocessing
