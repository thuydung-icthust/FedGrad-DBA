import argparse
import json
import datetime
import os
import logging
import torch
import torch.nn as nn
import train
import test
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import csv
from torchvision import transforms
from loan_helper import LoanHelper
from image_helper import ImageHelper
from utils.utils import dict_html
import utils.csv_record as csv_record
import yaml
import time
import numpy as np
import random
import config
import copy
import os
import wandb 
from fedgrad_helper import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger("logger")
criterion = torch.nn.CrossEntropyLoss()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)


def trigger_test_byindex(helper, index, vis, epoch, device="cuda"):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test.Mytest_poison_trigger(helper=helper, model=helper.target_model,
                                   adver_trigger_index=index, device=device)
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_index_" + str(index) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])
    
    # Only for plot
    # if helper.params['vis_trigger_split_test']:
    #     helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc, loss=None,
    #                                                eid=helper.params['environment_name'],
    #                                                name="global_in_index_" + str(index) + "_trigger")
def trigger_test_byname(helper, agent_name_key, vis, epoch, device="cuda"):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test.Mytest_poison_agent_trigger(helper=helper, model=helper.target_model, agent_name_key=agent_name_key, device=device)
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_" + str(agent_name_key) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])
    return epoch_loss, epoch_acc, epoch_corret, epoch_total
    # if helper.params['vis_trigger_split_test']:
        # helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc, loss=None,
        #                                            eid=helper.params['environment_name'],
        #                                            name="global_in_" + str(agent_name_key) + "_trigger")

if __name__ == '__main__':
    print('Start training')
    
    np.random.seed(7)
    time_start_load_everything = time.time()
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()
    with open(f'./{args.params}', 'r') as f:
        print(f"F here: {f}")
        params_loaded = yaml.safe_load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

    # if not os.path.exists(f'{params_loaded['log_folder']}/{args.wandb_group}'):
    #     os.makedirs(f'{args.log_folder}/{args.wandb_group}')
    group_name = params_loaded['wandb_group']
    instance_name = params_loaded['wandb_instance']
    defense = params_loaded['defense']
    defender =  None
    device = params_loaded['device']
    centralized_attack = params_loaded['centralized_attack'] # Centralized or distributed attack
    constrain = params_loaded['constrain'] and True
    wandb_ins = wandb.init(project="Backdoor attack in FL",
                            entity="yourentity",
                            name=instance_name,
                            group=group_name)
    poison_ratio = params_loaded['poison_ratio']
    part_nets_per_round = params_loaded['no_models']
    total_participants = params_loaded['number_of_total_participants']
    # pdg_attack = params_loaded['pdg_attack']
    total_attackers = int(poison_ratio*total_participants)
    adversary_idxs = np.random.choice(total_participants, total_attackers, replace=False)
    if not params_loaded['is_random_adversary']:
        adversary_idxs = params_loaded['adversary_list']
    print(f"adversary_idxs: {adversary_idxs}")
    if params_loaded['type'] == config.TYPE_LOAN:
        helper = LoanHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'loan'), device=device)
        helper.load_data(params_loaded)
    elif params_loaded['type'] == config.TYPE_CIFAR:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'cifar'),
                             device=device, centralized_attack=centralized_attack,
                             adversary_idxs=adversary_idxs, total_poisoned_batch=params_loaded['total_poisoned_batch'])
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_MNIST:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'mnist'), 
                             device=device, centralized_attack=centralized_attack,
                             adversary_idxs=adversary_idxs, 
                             total_poisoned_batch=params_loaded['total_poisoned_batch'])
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_TINYIMAGENET:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'tiny'), device=device)
        helper.load_data()
    else:
        helper = None
    if defense == 'fedgrad':
        num_adv = int(0.25*helper.params['no_models'])
        defender = FedGrad(total_workers = total_participants, 
                           num_workers = helper.params['no_models'], 
                           num_adv = num_adv, 
                           num_valid = 1, instance="benchmark", 
                           use_trustworthy=True)
    logger.info(f'load data done')
    helper.create_model()
    logger.info(f'create model done')
    ### Create models
    # if helper.params['is_poison']:
    #     logger.info(f"Poisoned following participants: {(helper.params['adversary_list'])}")

    best_loss = float('inf')

    # vis.text(text=dict_html(helper.params, current_time=helper.params["current_time"]),
    #          env=helper.params['environment_name'], opts=dict(width=300, height=400))
    logger.info(f"We use following environment for graphs:  {helper.params['environment_name']}")

    weight_accumulator = helper.init_weight_accumulator(helper.target_model)

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)

    submit_update_dict = None
    num_no_progress = 0
    old_w_accumulator = weight_accumulator
    
    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1, helper.params['aggr_epoch_interval']):
        start_time = time.time()
        t = time.time()

        agent_name_keys = helper.participants_list
        adversarial_name_keys = []
        net_list = []
        selected_adversary_idxs = []
        round_adversary_idxs = []
        if helper.params['is_random_namelist']:
            if helper.params['is_random_adversary']:  # random choose , maybe don't have advasarial
                # Our scenario is this case
                agent_name_keys = random.sample(helper.participants_list, helper.params['no_models'])
                for _id, _name_keys in enumerate(agent_name_keys):
                    # if _name_keys in helper.params['adversary_list']:
                    #     adversarial_name_keys.append(_name_key
                    if _name_keys in adversary_idxs:
                        adversarial_name_keys.append(_name_keys)
                        selected_adversary_idxs.append(_name_keys)
                        round_adversary_idxs.append(_id)
            else:  # must have advasarial if this epoch is in their poison epoch
                # ongoing_epochs = list(range(epoch, epoch + helper.params['aggr_epoch_interval']))
                for idx in range(0, len(helper.params['adversary_list'])):
                    print(f"idx: {idx}")
                    print(f"adversarial_name_keys: {adversarial_name_keys}")
                    # for ongoing_epoch in ongoing_epochs:
                    # if ongoing_epoch in helper.params[str(idx) + '_poison_epochs']:
                    if helper.params['adversary_list'][idx] not in adversarial_name_keys:
                        adversarial_name_keys.append(helper.params['adversary_list'][idx])
                        selected_adversary_idxs.append(helper.params['adversary_list'][idx])
                        round_adversary_idxs.append(idx)

                nonattacker=[]
                for adv in helper.params['adversary_list']:
                    if adv not in adversarial_name_keys:
                        nonattacker.append(copy.deepcopy(adv))
                benign_num = helper.params['no_models'] - len(adversarial_name_keys)
                random_agent_name_keys = random.sample(helper.benign_namelist+nonattacker, benign_num)
                agent_name_keys = adversarial_name_keys + random_agent_name_keys
        else:
            if helper.params['is_random_adversary']==False:
                # adversarial_name_keys=copy.deepcopy(helper.params['adversary_list'])
                adversarial_name_keys=copy.deepcopy(adversary_idxs)
        # print(f"self.model: {helper.target_model}")
        logger.info(f'Server comm. round: {epoch} choose agents : {agent_name_keys}.')
        print(f"Selected adversary idxs for this epoch is: {selected_adversary_idxs}.")
        # print(f"Round adversary idxs for this epoch is: {round_adversary_idxs}.")
        epochs_submit_update_dict, num_samples_dict, net_list = train.train(helper=helper, start_epoch=epoch,
                                                                  local_model=helper.local_model,
                                                                  target_model=helper.target_model,
                                                                  is_poison=helper.params['is_poison'],
                                                                  agent_name_keys=agent_name_keys, 
                                                                  adversary_idxs=selected_adversary_idxs,
                                                                  device=device,
                                                                  centralized_attack=centralized_attack,
                                                                  constrain=constrain,
                                                                  g_epc=epoch
                                                                  )
        logger.info(f'time spent on training: {time.time() - t}')
        print(f"Round adversary idxs for this epoch is: {round_adversary_idxs}.")
        cp_epochs_submit_update_dict = copy.deepcopy(epochs_submit_update_dict)
        cp_weight_accumulator = copy.deepcopy(weight_accumulator)
        weight_accumulator, updates = helper.accumulate_weight(weight_accumulator, epochs_submit_update_dict,
                                                               agent_name_keys, num_samples_dict)
        is_updated = True
        # print(f"before: epochs_submit_update_dict: {epochs_submit_update_dict}")
        if defender:
            pseudo_avg_net = copy.deepcopy(helper.target_model).to(device)
            helper.average_shrink_models(weight_accumulator, pseudo_avg_net, 10, device, part_nets_per_round) # get the pseudo avg models
            model_name = "MnistNet" if params_loaded['type'] == 'mnist' else "ResNet18"
            selected_net_indx, reconstructed_freq = defender.exec(client_models = net_list, 
                          num_dps = num_samples_dict, 
                          net_freq = [], 
                          net_avg = helper.target_model, 
                          g_user_indices = agent_name_keys, 
                          pseudo_avg_net = pseudo_avg_net, 
                          round = epoch, 
                          selected_attackers = selected_adversary_idxs, 
                          model_name = model_name, 
                          device=device)

            # if not selected_net_indx:
            #     break
                # cp_weight_accumulator, cp_updates = helper.accumulate_weight(cp_weight_accumulator, epochs_submit_update_dict,
                                                            #    agent_name_keys, num_samples_dict)
                # helper.average_shrink_models(old_w_accumulator, helper.target_model, 10, device)
            if selected_net_indx:
                # print(f"after: epochs_submit_update_dict: {epochs_submit_update_dict['99']}")
                cp_weight_accumulator, updates = helper.accumulate_weight(cp_weight_accumulator, cp_epochs_submit_update_dict, agent_name_keys, num_samples_dict, selected_net_indx, reconstructed_freq)    
                helper.average_shrink_models(cp_weight_accumulator, helper.target_model, 10, device, len(selected_net_indx))
            # helper.target_model = neo_net_list[0]
        elif helper.params['aggregation_methods'] == config.AGGR_MEAN:
            # Average the models
            is_updated = helper.average_shrink_models(weight_accumulator=weight_accumulator,
                                                      target_model=helper.target_model,
                                                      epoch_interval=helper.params['aggr_epoch_interval'], 
                                                      device=device,
                                                      no_models=part_nets_per_round)
            num_oracle_calls = 1
        elif helper.params['aggregation_methods'] == config.AGGR_GEO_MED:
            maxiter = helper.params['geom_median_maxiter']
            num_oracle_calls, is_updated, names, weights, alphas = helper.geometric_median_update(helper.target_model, updates, maxiter=maxiter)
            # vis_agg_weight(helper, names, weights, epoch, vis, adversarial_name_keys)
            # vis_fg_alpha(helper, names, alphas, epoch, vis, adversarial_name_keys)

        elif helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
            is_updated, names, weights, alphas = helper.foolsgold_update(helper.target_model, updates)
            # vis_agg_weight(helper,names,weights,epoch,vis,adversarial_name_keys)
            # vis_fg_alpha(helper,names,alphas,epoch,vis,adversarial_name_keys )
            num_oracle_calls = 1

        # clear the weight_accumulator
        old_w_accumulator = copy.deepcopy(weight_accumulator)
        weight_accumulator = helper.init_weight_accumulator(helper.target_model)
        temp_global_epoch = epoch + helper.params['aggr_epoch_interval'] - 1
        epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=temp_global_epoch,
                                                                       model=helper.target_model, device = device, is_poison=False,
                                                                       visualize=True, agent_name_key="global")
        print(f"epoch_acc: {epoch_acc}")
        csv_record.test_result.append(["global", temp_global_epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])
        if len(csv_record.scale_temp_one_row)>0:
            csv_record.scale_temp_one_row.append(round(epoch_acc, 4))
        poison_epoch_loss, poison_epoch_acc, poison_epoch_corret, poison_epoch_total = 0,0,0,0
        if helper.params['is_poison']:

            epoch_loss_p, epoch_acc_p, epoch_corret_p, epoch_total_p = test.Mytest_poison(helper=helper,
                                                                                    epoch=temp_global_epoch,
                                                                                    model=helper.target_model,
                                                                                    device = device,
                                                                                    is_poison=True,
                                                                                    visualize=True,
                                                                                    agent_name_key="global")
            csv_record.posiontest_result.append(
                ["global", temp_global_epoch, epoch_loss_p, epoch_acc_p, epoch_corret_p, epoch_total_p])


            # test on local triggers
            csv_record.poisontriggertest_result.append(
                ["global", "combine", "", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])
            # if helper.params['vis_trigger_split_test']:
            #     helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc_p, loss=None,
            #                                                eid=helper.params['environment_name'],
            #                                                name="global_combine")
            # if len(helper.params['adversary_list']) == 1:  # centralized attack
            if len(adversary_idxs) == 1:  # centralized attack
                if helper.params['centralized_test_trigger'] == True:  # centralized attack test on local triggers
                    for j in range(0, helper.params['trigger_num']):
                        trigger_test_byindex(helper, j, None, epoch, device)
            print(f"\nAt fl training round: {epoch}, MA = {epoch_acc}, BA = {epoch_acc_p}.")            
            # else:  # distributed attack
            #     for agent_name_key in adversary_idxs:
            #         poison_epoch_loss, poison_epoch_acc, poison_epoch_corret, poison_epoch_total = trigger_test_byname(helper, agent_name_key, None, epoch)

        wandb_logging_items = {
            "round": epoch,
            "epoch_loss": epoch_loss, 
            "maintask_acc_epoch": epoch_acc, 
            "epoch_corret": epoch_corret, 
            "epoch_total": epoch_total, 
            "poison_epoch_loss": epoch_loss_p, 
            "backdoor_acc_epoch": epoch_acc_p, 
            "poison_epoch_corret": epoch_corret_p, 
            "poison_epoch_total": epoch_total,
        }
        wandb_ins.log({"general": wandb_logging_items})
        helper.save_model(epoch=epoch, val_loss=epoch_loss)
        logger.info(f'Done in {time.time() - start_time} sec.')
        csv_record.save_result_csv(epoch, helper.params['is_poison'], helper.folder_path)



    logger.info('Saving all the graphs.')
    logger.info(f"This run has a label: {helper.params['current_time']}. "
                f"Visdom environment: {helper.params['environment_name']}")


    # vis.save([helper.params['environment_name']])
