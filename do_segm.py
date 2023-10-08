#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT

import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, indicator=["configs"])
import hydra

import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import (CityscapesDataset, CrossCityDataset, get_test_transforms,
					  get_train_transforms)
from generate_pseudo_labels import validate_model
from utils import (ScoreUpdater, adjust_learning_rate, cleanup,
				   get_arguments, label_selection, parse_split_list,
				   savelst_tgt, seed_torch, self_training_regularized_infomax,
				   self_training_regularized_infomax_cct, set_logger)
from datasets import CrossCityDataset, get_val_transforms
from tqdm import tqdm
from PIL import Image
from utils import ScoreUpdater, colorize_mask
import logging

def validate_model(model, save_round_eval_path, round_idx, args):
	logger = logging.getLogger('crosscityadap')
	## Doubles as a pseudo label generator
	osp = os.path
	val_transforms = get_val_transforms(args)
	if args.city != 'cityscapes':
		dataset = CrossCityDataset(root=args.data_tgt_dir, list_path=args.data_tgt_train_list.format(args.city), transforms=val_transforms)

	else:
		dataset = CityscapesDataset(
			# pseudo_root=save_pseudo_label_path, 
			list_path='./datasets/city_list/val.txt',
			transforms=val_transforms,
			debug=args.debug)
	loader = DataLoader(dataset, batch_size=12, num_workers=4, pin_memory=torch.cuda.is_available())

	scorer = ScoreUpdater(args.num_classes, len(loader))

	save_pred_vis_path = osp.join(save_round_eval_path, 'pred_vis')
	save_prob_path = osp.join(save_round_eval_path, 'prob')
	save_pred_path = osp.join(save_round_eval_path, 'pred')
	if not os.path.exists(save_pred_vis_path):
		os.makedirs(save_pred_vis_path)
	if not os.path.exists(save_prob_path):
		os.makedirs(save_prob_path)
	if not os.path.exists(save_pred_path):
		os.makedirs(save_pred_path)

	conf_dict = {k: [] for k in range(args.num_classes)}
	pred_cls_num = np.zeros(args.num_classes)
	## evaluation process
	logger.info('###### Start evaluating target domain train set in round {}! ######'.format(round_idx))
	start_eval = time.time()
	model.eval()
	with torch.no_grad():
		for batch in tqdm(loader):
			image, label, name = batch

			image = image.to(args.device)
			output = model(image).cpu().softmax(1)

			flipped_out = model(image.flip(-1)).cpu().softmax(1)
			output = 0.5 * (output + flipped_out.flip(-1))

			# image = image.cpu()
			pred_prob, pred_labels = output.max(1)
			# scorer.update(pred_labels.view(-1), label.view(-1))

			for b_ind in range(image.size(0)):
				image_name = name[b_ind].split('/')[-1].split('.')[0]

				np.save('%s/%s.npy' % (save_prob_path, image_name), output[b_ind].numpy().transpose(1, 2, 0))
				if args.debug:
					colorize_mask(pred_labels[b_ind].numpy().astype(np.uint8)).save(
						'%s/%s_color.png' % (save_pred_vis_path, image_name))
				Image.fromarray(pred_labels[b_ind].numpy().astype(np.uint8)).save(
					'%s/%s.png' % (save_pred_path, image_name))

			if args.kc_value == 'conf':
				for idx_cls in range(args.num_classes):
					idx_temp = pred_labels == idx_cls
					pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + idx_temp.sum()
					if idx_temp.any():
						conf_cls_temp = pred_prob[idx_temp].numpy().astype(np.float32)[::args.ds_rate]
						conf_dict[idx_cls].extend(conf_cls_temp)
	model.train()
	logger.info('###### Finish evaluating target domain train set in round {}! Time cost: {:.2f} seconds. ######'.format(
		round_idx, time.time() - start_eval))
	return conf_dict, pred_cls_num, save_prob_path, save_pred_path

def test(model, round_idx, args, logger):
	transforms = get_test_transforms()
	
	if args.city != 'cityscapes':
		ds = CrossCityDataset(root=args.data_tgt_dir, list_path=args.data_tgt_test_list.format(args.city), transforms=transforms)
		# tgtds = CrossCityDataset(args.data_tgt_dir.format(args.city), tgt_train_lst,
		#                         pseudo_root=save_pseudo_label_path, transforms=tgt_transforms)
	else:
		# ds = CrossCityDataset(root=args.data_tgt_dir, list_path=args.data_tgt_test_list.format(args.city), transforms=transforms)
		ds = CityscapesDataset(
			# pseudo_root=save_pseudo_label_path, 
			list_path='./datasets/city_list/val.txt',
			transforms=transforms, debug=args.debug)
		
	loader = torch.utils.data.DataLoader(ds, batch_size=6, pin_memory=torch.cuda.is_available(), num_workers=6)

	scorer = ScoreUpdater(args.num_classes, len(loader))
	logger.info('###### Start evaluating in target domain test set in round {}! ######'.format(round_idx))
	start_eval = time.time()
	model.eval()
	with torch.no_grad():
		for batch in loader:
			img, label, _ = batch
			pred = model(img.to(args.device)).argmax(1).cpu()
			scorer.update(pred.view(-1), label.view(-1))
	model.train()
	logger.info('###### Finish evaluating in target domain test set in round {}! Time cost: {:.2f} seconds. ######'.format(
		round_idx, time.time()-start_eval))
	return scorer.scores()

def train(mix_trainloader, model, interp, optimizer, args, logger):
	"""Create the model and start the training."""
	tot_iter = len(mix_trainloader)
	for i_iter, batch in enumerate(mix_trainloader):
		images, labels, name = batch
		labels = labels.long()

		optimizer.zero_grad()
		adjust_learning_rate(optimizer, i_iter, tot_iter, args)

		if args.info_max_loss:
			pred = model(images.to(args.device), training=True)
			loss = self_training_regularized_infomax(pred, labels.to(args.device), args)
		elif args.unc_noise:
			pred, noise_pred = model(images.to(args.device), training=True)
			loss = self_training_regularized_infomax_cct(pred, labels.to(args.device), noise_pred, args)
		else:
			pred = model(images.to(args.device))
			loss = F.cross_entropy(pred, labels.to(args.device), ignore_index=255)

		loss.backward()
		optimizer.step()

		logger.info('iter = {} of {} completed, loss = {:.4f}'.format(i_iter+1, tot_iter, loss.item()))


def config_format(cfg):
	"""Formats config to be saved to wandb."""
	from pytorch_lightning.utilities.logger import _convert_params, _flatten_dict, _sanitize_callable_params
	params = _convert_params(_flatten_dict(_sanitize_callable_params(cfg)))
	return params

def print_config_tree(
	cfg,
	print_order = (
		"task_name", 
		"tags", 
		"env", 
		"net",
		"policy", 
		"optimizer", 
		"train_collector", 
		"test_collector",
		"trainer",
	),
	resolve: bool = False,
	save_to_file: bool = False,
	) -> None:
	"""Prints content of DictConfig using Rich library and its tree structure.

	Args:
		cfg (DictConfig): Configuration composed by Hydra.
		print_order (Sequence[str], optional): Determines in what order config components are printed.
		resolve (bool, optional): Whether to resolve reference fields of DictConfig.
		save_to_file (bool, optional): Whether to export config to the hydra output folder.
	"""
	from omegaconf import DictConfig, OmegaConf
	import rich
	import rich.syntax
	import rich.tree
	style = "dim"
	tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

	queue = []

	# add fields from `print_order` to queue
	for field in print_order:
		queue.append(field) if field in cfg else None

	# add all the other fields to queue (not specified in `print_order`)
	for field in cfg:
		if field not in queue:
			queue.append(field)

	# generate config tree from queue
	for field in queue:
		branch = tree.add(field, style=style, guide_style=style)
		config_group = cfg[field]
		if isinstance(config_group, DictConfig):
			branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
		else:
			branch_content = str(config_group)

		branch.add(rich.syntax.Syntax(branch_content, "yaml"))

	# print config tree
	rich.print(tree)

	# save config tree to file
	if save_to_file:
		with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
			rich.print(tree, file=file)

@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="entry.yaml")	
def main(args):
	# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = "cuda"
	osp = os.path

	print("\n\n\n### Printing Hydra config ...")
	print_config_tree(args, resolve=True)

	import wandb
	wandb.init(
		project=args.task_name,
		tags=args.tags,
		config=config_format(args),
		dir=args.output_dir,
		mode=args.wandb.mode,
		id = args.output_dir.replace("/","_")
	)


	# args = get_arguments()
	if not os.path.exists(args.save):
		os.makedirs(args.save)
	logger = set_logger(args.save, 'training_logger', False)

	num_classes = 19 if (
		"GTA" in args.restore_from or "SYNTHIA" in args.restore_from \
	) else 13
	args.num_classes = num_classes
	args.device = device



	seed_torch(args.randseed)

	logger.info('Starting training with arguments')
	logger.info(vars(args))

	save_path = args.save
	save_pseudo_label_path = osp.join(save_path, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
	save_stats_path = osp.join(save_path, 'stats')  # in 'save_path'
	save_lst_path = osp.join(save_path, 'list')
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	if not os.path.exists(save_pseudo_label_path):
		os.makedirs(save_pseudo_label_path)
	if not os.path.exists(save_stats_path):
		os.makedirs(save_stats_path)
	if not os.path.exists(save_lst_path):
		os.makedirs(save_lst_path)

	tgt_portion = args.init_tgt_port
	if args.city != 'cityscapes':
		image_tgt_list, image_name_tgt_list, _, _ = parse_split_list(args.data_tgt_train_list.format(args.city))

	model = make_network(args).to(device)
	# test(model, -1)
	for round_idx in range(args.num_rounds):
		save_round_eval_path = osp.join(args.save, str(round_idx))
		save_pseudo_label_color_path = osp.join(
			save_round_eval_path, 'pseudo_label_color')  # in every 'save_round_eval_path'
		if not os.path.exists(save_round_eval_path):
			os.makedirs(save_round_eval_path)
		if not os.path.exists(save_pseudo_label_color_path):
			os.makedirs(save_pseudo_label_color_path)
		src_portion = args.init_src_port
		########## pseudo-label generation
		conf_dict, pred_cls_num, save_prob_path, save_pred_path = validate_model(model,
																				 save_round_eval_path,
																				 round_idx, args)
		cls_thresh = label_selection.kc_parameters(
			conf_dict, pred_cls_num, tgt_portion, round_idx, save_stats_path, args)

		label_selection.label_selection(cls_thresh, round_idx, save_prob_path, save_pred_path,
										save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path, args)

		tgt_portion = min(tgt_portion + args.tgt_port_step, args.max_tgt_port)
		if args.city != 'cityscapes':
			tgt_train_lst = savelst_tgt(image_tgt_list, image_name_tgt_list, save_lst_path, save_pseudo_label_path)

		rare_id = np.load(save_stats_path + '/rare_id_round' + str(round_idx) + '.npy')
		mine_id = np.load(save_stats_path + '/mine_id_round' + str(round_idx) + '.npy') #  # ! mine id does not used later
		# mine_chance = args.mine_chance

		src_transforms, tgt_transforms = get_train_transforms(args, mine_id)
		srcds = CityscapesDataset(transforms=src_transforms, debug=args.debug)

		if args.city != 'cityscapes':
			tgtds = CrossCityDataset(args.data_tgt_dir.format(args.city), tgt_train_lst,
									  pseudo_root=save_pseudo_label_path, transforms=tgt_transforms)
		else:
			tgtds = CityscapesDataset(
				pseudo_root=save_pseudo_label_path, 
				list_path='./datasets/city_list/val.txt',
				transforms=tgt_transforms,
				debug=args.debug)
		
		if args.no_src_data: # ! ? TODO
			mixtrainset = tgtds
		else:
			raise ValueError("We only consider no source data setting.")
			mixtrainset = torch.utils.data.ConcatDataset([srcds, tgtds])

		mix_loader = DataLoader(mixtrainset, batch_size=args.batch_size, shuffle=True,
								num_workers=args.batch_size, pin_memory=torch.cuda.is_available())
		src_portion = min(src_portion + args.src_port_step, args.max_src_port)
		optimizer = optim.SGD(model.optim_parameters(args), lr=args.learning_rate,
							  momentum=args.momentum, weight_decay=args.weight_decay)
		interp = nn.Upsample(size=args.input_size[::-1], mode='bilinear', align_corners=True)
		torch.backends.cudnn.enabled = True  # enable cudnn
		torch.backends.cudnn.benchmark = True
		start = time.time()
		for epoch in range(args.epr):
			train(mix_loader, model, interp, optimizer, args, logger)
			print('taking snapshot ...')
			torch.save(model.state_dict(), osp.join(args.save,
													'2nthy_round' + str(round_idx) + '_epoch' + str(epoch) + '.pth'))
		end = time.time()
		
		logger.info('###### Finish model retraining dataset in round {}! Time cost: {:.2f} seconds. ######'.format(
			round_idx, end - start))
		mious = test(model, round_idx, args, logger)
		wandb.log({"miou_round": mious.mean()})
		cleanup(args.save)
	cleanup(args.save)
	shutil.rmtree(save_pseudo_label_path)
	mious = test(model, args.num_rounds - 1, args, logger)
	wandb.log({"miou_final": mious.mean()})


if __name__ == "__main__":
	main()
