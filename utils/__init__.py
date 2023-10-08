#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT

from . import label_selection
from .argparser import get_arguments
from .code_utils import cleanup, seed_torch
from .eval import ScoreUpdater
from .list_utils import parse_split_list, savelst_srctgt, savelst_tgt
from .logger_utils import np_print_options, set_logger
from .loss import (self_training_regularized_infomax,
                   self_training_regularized_infomax_cct)
from .lr_utils import adjust_learning_rate
from .viz_utils import colorize_mask

from network import JointSegAuxDecoderModel, NoisyDecoders
import torch
from network import DeeplabMulti as DeepLab

def make_network(args):
	model = DeepLab(args.num_classes, False)
	model = torch.nn.DataParallel(model)
	rf = torch.load(args.restore_from, map_location=args.device)
	sd = rf['state_dict'] if 'state_dict' in rf else rf
	if 'state_dict' in rf:
		sd = rf['state_dict']
	else:
		sd = rf
		sd = {"module."+k: v for k, v in sd.items()}
	# add prefix 'module.' to every key
	model.load_state_dict(sd)

	model = model.module
	if args.unc_noise:
		aux_decoders = NoisyDecoders(args.decoders, args.dropout, args.num_classes)
		model = JointSegAuxDecoderModel(model, aux_decoders)
	return model
