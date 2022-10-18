import torch 
import time
import numpy as np
from numpy import dot
from numpy.linalg import norm

import dba
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
from sklearn.cluster import KMeans

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])

def min_max_scale(data_r):
    data_r = np.asarray(data_r)
    v = data_r[:].reshape((-1,1))
    v_scaled = min_max_scaler.fit_transform(v)
    data_r = v_scaled
    return data_r

def load_model_weight(net, weight):
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data =  weight[index_bias:index_bias+p.numel()].view(p.size())
        index_bias += p.numel()

# def get_logging_items(net_list, additional_net, custom_net_2, selected_node_indices, avg_net_prev, avg_net, attackers_idxs, fl_round):
#     logging_list = []
#     recorded_w_list = []
#     recorded_w_list.append(vectorize_net(additional_net))
    
#     for cm in net_list:
#         recorded_w_list.append(vectorize_net(cm))    
    
#     for i,param in enumerate(additional_net.classifier.parameters()):
#         if i == 0:
#             with open('logging/w_benchmark_01_200.csv', 'a+') as w_f:
#                 write = csv.writer(w_f)
#                 write.writerow(param.data.cpu().numpy())
#     additional_item = [fl_round, 0, -3, list(additional_net.classifier.parameters())[1].data.cpu().numpy()]
#     logging_list.append(additional_item)
    
#     #CUSTOM NET 2
#     for i,param in enumerate(custom_net_2.classifier.parameters()):
#         if i == 0:
#             with open('logging/w_benchmark_01_200.csv', 'a+') as w_f:
#                 write = csv.writer(w_f)
#                 write.writerow(param.data.cpu().numpy())
#     additional_item_2 = [fl_round, 0, -4, list(custom_net_2.classifier.parameters())[1].data.cpu().numpy()]
#     logging_list.append(additional_item_2)
    
#     for net_idx, global_user_idx in enumerate(selected_node_indices):
#         #round id weights bias is-attacker
#         net = net_list[net_idx]
#         is_attacker = 0
#         # bias = list(net.classifier.parameters())[0].data.cpu().numpy()
#         # weights = list(net.classifier.parameters())[-1].data.cpu().numpy()

#         for idx, param in enumerate(net.classifier.parameters()):
#             if idx:
#                 bias = param.data.cpu().numpy()
#             else:
#                 weights = param.data.cpu().numpy()
#         # with open('logging/bias_benchmark.csv', 'a+') as bias_f:
#         #     write = csv.writer(bias_f)
#         #     write.writerow([bias])
#         with open('logging/w_benchmark_01_200.csv', 'a+') as w_f:
#             write = csv.writer(w_f)
#             write.writerow(weights)        
#             # write.writerow([weight])
#         if global_user_idx in attackers_idxs:
#             is_attacker = 1
#         item = [fl_round, is_attacker, global_user_idx, bias]
#         logging_list.append(item)
    
#     prev_avg_item = [fl_round, 0, -2, list(avg_net_prev.classifier.parameters())[1].data.cpu().numpy()] if avg_net_prev else [fl_round, 0, -2, None]
#     avg_item = [fl_round, 0, -1, list(avg_net.classifier.parameters())[1].data.cpu().numpy()]
    
#     recorded_w_list.append(vectorize_net(avg_net_prev))
#     recorded_w_list.append(vectorize_net(avg_net))

#     # with open('logging/flatten_w_benchmark.csv', 'a+') as w_f:
#     #     write = csv.writer(w_f)
#     #     for item_w in recorded_w_list:
#     #         write.writerow(item_w)    
                
#     for i,param in enumerate(avg_net_prev.classifier.parameters()):
#         if i == 0:
#             with open('logging/w_benchmark_01_200.csv', 'a+') as w_f:
#                 write = csv.writer(w_f)
#                 write.writerow(param.data.cpu().numpy())    
#     for i,param in enumerate(avg_net.classifier.parameters()):
#         if i == 0:
#             with open('logging/w_benchmark_01_200.csv', 'a+') as w_f:
#                 write = csv.writer(w_f)
#                 write.writerow(param.data.cpu().numpy())        
#     logging_list.append(prev_avg_item)
#     logging_list.append(avg_item)
#     return logging_list

# def get_logging_items_full_w(net_list, additional_net, custom_net_2, selected_node_indices, avg_net_prev, avg_net, attackers_idxs, fl_round):
#     logging_list = []
#     recorded_w_list = []
#     print(f'[Dung] net_list_len: {len(net_list)}')
#     recorded_w_list.append(additional_net.state_dict())
#     recorded_w_list.append(custom_net_2.state_dict())
#     for cm in net_list:
#         recorded_w_list.append(cm.state_dict())
#     recorded_w_list.append(avg_net_prev.state_dict())
#     recorded_w_list.append(avg_net.state_dict())

#     ids = [-3, -4, *selected_node_indices, -2, -1]

#     if not os.path.exists('logging/eps10_400'):
#         os.makedirs('logging/eps10_400')
#     for i, idx in enumerate(ids):
#         torch.save(recorded_w_list[i], open(f'logging/eps10_400/{idx}_net.pth', 'wb'))
#     # for i,param in enumerate(additional_net.classifier.parameters()):
#     #     if i == 0:
#     #         with open('logging/weight_benchmark_01.csv', 'a+') as w_f:
#     #             write = csv.writer(w_f)
#     #             write.writerow(param.data.cpu().numpy())
#     additional_item = [fl_round, 0, -3, list(additional_net.classifier.parameters())[1].data.cpu().numpy()]
#     logging_list.append(additional_item)
#     additional_item_2 = [fl_round, 0, -4, list(custom_net_2.classifier.parameters())[1].data.cpu().numpy()]
#     logging_list.append(additional_item_2)
#     for net_idx, global_user_idx in enumerate(selected_node_indices):
#         #round id weights bias is-attacker
#         net = net_list[net_idx]
#         is_attacker = 0
#         # bias = list(net.classifier.parameters())[0].data.cpu().numpy()
#         # weights = list(net.classifier.parameters())[-1].data.cpu().numpy()

#         for idx, param in enumerate(net.classifier.parameters()):
#             if idx:
#                 bias = param.data.cpu().numpy()
#             else:
#                 weights = param.data.cpu().numpy()
#         # with open('logging/bias_benchmark.csv', 'a+') as bias_f:
#         #     write = csv.writer(bias_f)
#         #     write.writerow([bias])
#         # with open('logging/weight_benchmark_01.csv', 'a+') as w_f:
#         #     write = csv.writer(w_f)
#         #     write.writerow(weights)        
#             # write.writerow([weight])
#         if global_user_idx in attackers_idxs:
#             is_attacker = 1
#         item = [fl_round, is_attacker, global_user_idx, bias]
#         logging_list.append(item)
    
#     prev_avg_item = [fl_round, 0, -2, list(avg_net_prev.classifier.parameters())[1].data.cpu().numpy()] if avg_net_prev else [fl_round, 0, -2, None]
#     avg_item = [fl_round, 0, -1, list(avg_net.classifier.parameters())[1].data.cpu().numpy()]
    
    

#     # with open('logging/flatten_w_benchmark.csv', 'a+') as w_f:
#     #     write = csv.writer(w_f)
#     #     for item_w in recorded_w_list:
#     #         write.writerow(item_w)    
                
#     # for i,param in enumerate(avg_net_prev.classifier.parameters()):
#     #     if i == 0:
#     #         with open('logging/weight_benchmark_01.csv', 'a+') as w_f:
#     #             write = csv.writer(w_f)
#     #             write.writerow(param.data.cpu().numpy())    
#     # for i,param in enumerate(avg_net.classifier.parameters()):
#     #     if i == 0:
#     #         with open('logging/weight_benchmark_01.csv', 'a+') as w_f:
#     #             write = csv.writer(w_f)
#     #             write.writerow(param.data.cpu().numpy())        
#     logging_list.append(prev_avg_item)
#     logging_list.append(avg_item)
#     return logging_list
         
# def get_logging_items_new(net_list, selected_node_indices, avg_net_prev, avg_net, exploration_net, g_attackers_idxs, fl_round):
#     logging_list = []
    
#     for net_idx, global_user_idx in enumerate(selected_node_indices):
#         net = net_list[net_idx]
#         is_attacker = 0
#         w_log_file_name = 'logging/attacker_weight.csv' if global_user_idx in g_attackers_idxs else 'logging/normal_weight.csv'
        
#         # Different log for attackers
#         bias, weight = None, None
#         for idx, param in enumerate(net.classifier.parameters()):
#             if idx:
#                 bias = param.data.cpu().numpy()
#             else:
#                 weight = param.data.cpu().numpy()
#         with open(w_log_file_name, 'a+') as w_f:
#             write = csv.writer(w_f)
#             write.writerow(weight)   
        
#         #round id weights bias is-attacker
#         if global_user_idx in g_attackers_idxs:
#             is_attacker = 1
#         item = [fl_round, is_attacker, global_user_idx, bias]
#         logging_list.append(item)
#     prev_avg_item = [fl_round, 0, -2, list(avg_net_prev.classifier.parameters())[1].data.cpu().numpy()] if avg_net_prev else [fl_round, 0, -2, None]
#     avg_item = [fl_round, 0, -1, list(avg_net.classifier.parameters())[1].data.cpu().numpy()]
    
#     for i,param in enumerate(avg_net_prev.classifier.parameters()):
#         if i == 0:
#             with open('logging/normal_weight.csv', 'a+') as w_f:
#                 write = csv.writer(w_f)
#                 write.writerow(param.data.cpu().numpy())    
#     for i,param in enumerate(avg_net.classifier.parameters()):
#         if i == 0:
#             with open('logging/normal_weight.csv', 'a+') as w_f:
#                 write = csv.writer(w_f)
#                 write.writerow(param.data.cpu().numpy())        
#     logging_list.append(prev_avg_item)
#     logging_list.append(avg_item)
#     return logging_list

def calculate_sum_grad_diff(meta_data, num_cli=11, num_w=512, glob_update=None):
    v_x = [num_w * i for i in range(num_cli)]
    total_label = 10
    sum_diff_by_label = []
    glob_temp_sum = None
    glob_ret = []
    print(f"num_w: {num_w}")
    
    for data in meta_data:
        data = data.flatten()
        ret = []
        for i in range(total_label):
            temp_sum = np.sum(data[v_x[i]:v_x[i+1]])
            ret.append(temp_sum)
        sum_diff_by_label.append(ret)
    if glob_update is not None:
        glob_update = glob_update.flatten()
        for i in range(total_label):
            glob_temp_sum = np.sum(glob_update[v_x[i]:v_x[i+1]])
            glob_ret.append(glob_temp_sum)
    return np.asarray(sum_diff_by_label), np.asarray(glob_ret)

def get_distance_on_avg_net(weight_list, avg_weight, weight_update, total_cli = 10):
    eucl_dis = []
    cs_dis = []
    for i in range(total_cli):
        # euclidean distance btw weight updates
        point = weight_update[i].flatten().reshape(-1,1)
        base_p = avg_weight.flatten().reshape(-1,1)
        ds = point - base_p
        sum_sq = np.dot(ds.T, ds)
        eucl_dis.append(float(np.sqrt(sum_sq).flatten()))
    for i in range(total_cli):
        # cosine similarity
        point = weight_list[i].flatten()
        base_p = avg_weight.flatten()
        cs = dot(point, base_p)/(norm(point)*norm(base_p))
        cs_dis.append(float(cs.flatten()))
    return eucl_dis, cs_dis

def get_cs_on_base_net(weight_update, avg_weight, total_cli = 10):
    cs_list = []
    total_cli = len(weight_update)
    base_p = avg_weight.flatten()
    # print(f"base_p: {base_p}")
    for i in range(total_cli):
        point = weight_update[i].flatten()
        # print("point: ", point)
        cs = dot(point, base_p)/(norm(point)*norm(base_p))
        cs_list.append(float(cs.flatten()))
    return cs_list

def get_ed_on_base_net(weight_update, avg_weight, total_cli = 10):
    ed_list = []
    total_cli = len(weight_update)
    for i in range(total_cli):
        point = weight_update[i].flatten().reshape(-1,1)
        base_p = avg_weight.flatten().reshape(-1,1)
        ds = point - base_p
        sum_sq = np.dot(ds.T, ds)
        ed_list.append(float(np.sqrt(sum_sq).flatten()))
    return ed_list
   
def extract_classifier_layer(net_list, global_avg_net, prev_net, model="vgg9"):
    bias_list = []
    weight_list = []
    weight_update = []
    avg_bias = None
    avg_weight = None
    prev_avg_bias = None
    prev_avg_weight = None
    last_model_layer = "classifier" if model=="vgg9" else "fc3" 
    
    # print(f"global_avg_net: {global_avg_net}")
    # print(f"prev_net: {prev_net}")
    # print(f"model: {model}")
    # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in prev_net.state_dict():
    #     print(param_tensor, "\t", prev_net.state_dict()[param_tensor].size())
    print(f"model: {model}")
    if model == "vgg9":
        for idx, param in enumerate(global_avg_net.classifier.parameters()):
            if idx:
                avg_bias = param.data.cpu().numpy()
            else:
                avg_weight = param.data.cpu().numpy()

        for idx, param in enumerate(prev_net.classifier.parameters()):
            if idx:
                prev_avg_bias = param.data.cpu().numpy()
            else:
                prev_avg_weight = param.data.cpu().numpy()
        glob_update = avg_weight - prev_avg_weight
        for net in net_list:
            bias = None
            weight = None
            for idx, param in enumerate(net.classifier.parameters()):
                if idx:
                    bias = param.data.cpu().numpy()
                else:
                    weight = param.data.cpu().numpy()
            bias_list.append(bias)
            weight_list.append(weight)
            weight_update.append(weight-avg_weight)
    elif model == "lenet":
        for idx, param in enumerate(global_avg_net.fc2.parameters()):
            if idx:
                avg_bias = param.data.cpu().numpy()
            else:
                avg_weight = param.data.cpu().numpy()

        for idx, param in enumerate(prev_net.fc2.parameters()):
            if idx:
                prev_avg_bias = param.data.cpu().numpy()
            else:
                prev_avg_weight = param.data.cpu().numpy()
        # print(f"")
        glob_update = avg_weight - prev_avg_weight
        for net in net_list:
            bias = None
            weight = None
            for idx, param in enumerate(net.fc2.parameters()):
                if idx:
                    bias = param.data.cpu().numpy()
                else:
                    weight = param.data.cpu().numpy()
            bias_list.append(bias)
            weight_list.append(weight)
            weight_update.append(weight-avg_weight)
    elif model == "MnistNet":
        for idx, param in enumerate(global_avg_net.fc2.parameters()):
            if idx:
                avg_bias = param.data.cpu().numpy()
            else:
                avg_weight = param.data.cpu().numpy()

        for idx, param in enumerate(prev_net.fc2.parameters()):
            if idx:
                prev_avg_bias = param.data.cpu().numpy()
            else:
                prev_avg_weight = param.data.cpu().numpy()
                
        glob_update = avg_weight - prev_avg_weight
        for net in net_list:
            bias = None
            weight = None
            for idx, param in enumerate(net.fc2.parameters()):
                if idx:
                    bias = param.data.cpu().numpy()
                else:
                    weight = param.data.cpu().numpy()
            bias_list.append(bias)
            weight_list.append(weight)
            weight_update.append(weight-avg_weight)          
    elif model == "ResNet18":
        for idx, param in enumerate(global_avg_net.linear.parameters()):
            if idx:
                avg_bias = param.data.cpu().numpy()
            else:
                avg_weight = param.data.cpu().numpy()

        for idx, param in enumerate(prev_net.linear.parameters()):
            if idx:
                prev_avg_bias = param.data.cpu().numpy()
            else:
                prev_avg_weight = param.data.cpu().numpy()
        glob_update = avg_weight - prev_avg_weight
        for net in net_list:
            bias = None
            weight = None
            for idx, param in enumerate(net.linear.parameters()):
                if idx:
                    bias = param.data.cpu().numpy()
                else:
                    weight = param.data.cpu().numpy()
            bias_list.append(bias)
            weight_list.append(weight)
            weight_update.append(weight-avg_weight)
    
    return bias_list, weight_list, avg_bias, avg_weight, weight_update, glob_update, prev_avg_weight


class Defense:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def exec(self, client_model, *args, **kwargs):
        raise NotImplementedError()

class FedGrad(Defense):
    """
    FedGrad by DungNT
    """
    def __init__(self, total_workers, num_workers, num_adv, num_valid = 1, instance="benchmark", use_trustworthy=False, *args, **kwargs):
        self.num_valid = num_valid
        self.num_workers = num_workers
        self.s = num_adv
        self.instance = instance
        self.choosing_frequencies = {}
        self.accumulate_c_scores = {}
        self.use_trustworthy = use_trustworthy
        self.pairwise_w = np.zeros((total_workers+1, total_workers+1))
        self.pairwise_b = np.zeros((total_workers+1, total_workers+1))
        self.eta = 0.5 # this parameter could be changed
        self.switch_round = 50 # this parameter could be changed
        self.trustworthy_threshold = 0.75
        self.lambda_1 = 0.25
        self.lambda_2 = 1.0
        
        print("Starting performing FedGrad...")
        self.pairwise_choosing_frequencies = np.zeros((total_workers, total_workers))
        self.trustworthy_scores = [[0.5] for _ in range(total_workers+1)]

    def exec(self, client_models, num_dps, net_freq, net_avg, g_user_indices, pseudo_avg_net, round, selected_attackers, model_name, device, *args, **kwargs):
        start_fedgrad_t = time.time()*1000
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        neighbor_distances = []
        print("Starting performing FedGrad...")
        
        # SOFT FILTER
        layer1_start_t = time.time()*1000
        bias_list, weight_list, avg_bias, avg_weight, weight_update, glob_update, prev_avg_weight = extract_classifier_layer(client_models, pseudo_avg_net, net_avg, model_name)
        total_client = len(g_user_indices)
        
        raw_c_scores = self.get_compromising_scores(glob_update, weight_update)
        c_scores = []
        for idx, cli in enumerate(g_user_indices):
            # increase the frequency of the selected choosen clients
            self.choosing_frequencies[cli] = self.choosing_frequencies.get(cli, 0) + 1
            # update the accumulator
            self.accumulate_c_scores[cli] = ((self.choosing_frequencies[cli] - 1) / self.choosing_frequencies[cli]) * self.accumulate_c_scores.get(cli, 0) + (1 / self.choosing_frequencies[cli]) *  raw_c_scores[idx]
            c_scores.append(self.accumulate_c_scores[cli])
        
        c_scores = np.array(c_scores)
        epsilon_1 = min(self.eta, np.median(c_scores))
        
        
        participated_attackers = []
        for in_, id_ in enumerate(g_user_indices):
            if id_ in selected_attackers:
                participated_attackers.append(in_)
        
        suspicious_idxs_1 = [ind_ for ind_ in range(total_client) if c_scores[ind_] > epsilon_1]
        print("[Soft-filter] predicted suspicious set is:: ", suspicious_idxs_1)
        layer1_end_t = time.time()*1000
        layer1_inf_time = layer1_end_t-layer1_start_t
        print(f"Total computation time of the 1st layer is: {layer1_inf_time}")
        
        # HARD FILTER
        layer2_start_t = time.time()*1000
        round_pw_bias = np.zeros((total_client, total_client))
        round_pw_weight = np.zeros((total_client, total_client))
        
        sum_diff_by_label, glob_temp_sum_by_label = calculate_sum_grad_diff(meta_data = weight_update, num_w = weight_update[0].shape[-1], glob_update=glob_update)
        norm_bias_list = normalize(bias_list, axis=1)
        norm_grad_diff_list = normalize(sum_diff_by_label, axis=1)
        
        # UPDATE CUMULATIVE COSINE SIMILARITY 
        for i, g_i in enumerate(g_user_indices):
            distance = []
            for j, g_j in enumerate(g_user_indices):
                self.pairwise_choosing_frequencies[g_i][g_j] = self.pairwise_choosing_frequencies[g_i][g_j] + 1.0
                bias_p_i = norm_bias_list[i]
                bias_p_j = norm_bias_list[j]
                cs_1 = np.dot(bias_p_i, bias_p_j)/(np.linalg.norm(bias_p_i)*np.linalg.norm(bias_p_j))
                round_pw_bias[i][j] = cs_1.flatten()
                
                w_p_i = norm_grad_diff_list[i]
                w_p_j = norm_grad_diff_list[j]
                cs_2 = np.dot(w_p_i, w_p_j)/(np.linalg.norm(w_p_i)*np.linalg.norm(w_p_j))
                round_pw_weight[i][j] = cs_2.flatten()
       
        # compute closeness scores 
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i+1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(float(np.linalg.norm(g_i-g_j)**2)) # let's change this to pytorch version
            neighbor_distances.append(distance)

        nb_in_score = self.num_workers-self.s-2
        scores = []
        for i, g_i in enumerate(vectorize_nets):
            dists = []
            for j, g_j in enumerate(vectorize_nets):
                if j == i:
                    continue
                if j < i:
                    dists.append(neighbor_distances[j][i - j - 1])
                else:
                    dists.append(neighbor_distances[i][j - i - 1])

            # alternative to topk in pytorch and tensorflow
            topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
            scores.append(sum(np.take(dists, topk_ind)))
        trusted_index = scores.index(min(scores)) # ==> trusted client is the client whose smallest closeness score.

        scaler = MinMaxScaler()
        round_pw_bias = scaler.fit_transform(round_pw_bias)
        round_pw_weight = scaler.fit_transform(round_pw_weight)

        # update cumulative information
        for i, g_i in enumerate(g_user_indices):
            for j, g_j in enumerate(g_user_indices):
                freq_appear = self.pairwise_choosing_frequencies[g_i][g_j]
                self.pairwise_w[g_i][g_j] = (freq_appear - 1)/freq_appear*self.pairwise_w[g_i][g_j] +  1/freq_appear*round_pw_weight[i][j]
                self.pairwise_b[g_i][g_j] = (freq_appear - 1)/freq_appear*self.pairwise_b[g_i][g_j] +  1/freq_appear*round_pw_bias[i][j]
                
        
        # From now on, trusted_model contains the index base model treated as valid user.
        suspicious_idxs_2 = []
        saved_pairwise_sim = []
        layer2_inf_t = 0.0
        
        final_suspicious_idxs = suspicious_idxs_1 # temporarily assigned by the first filter
        # NOW CHECK FOR SWITCH ROUND
        # TODO: find dynamic threshold
        # STILL PERFORM HARD-FILTER to save the historical information about colluding property.
        cummulative_w = self.pairwise_w[np.ix_(g_user_indices, g_user_indices)]
        cummulative_b = self.pairwise_b[np.ix_(g_user_indices, g_user_indices)]
        
        saved_pairwise_sim = np.hstack((cummulative_w, cummulative_b))
        kmeans = KMeans(n_clusters = 2)
        pred_labels = kmeans.fit_predict(saved_pairwise_sim)
        trusted_cluster_idx = pred_labels[trusted_index] # assign cluster containing trusted client as benign cluster
        malicious_cluster_idx = 0 if trusted_cluster_idx == 1 else 1
        suspicious_idxs_2 = np.argwhere(np.asarray(pred_labels) == malicious_cluster_idx).flatten()
        
        print("[Hard-filter] predicted suspicious set is: ", suspicious_idxs_2)
        layer2_end_t = time.time()*1000
        layer2_inf_t = layer2_end_t-layer2_start_t
        print(f"Total computation time of the 2nd layer is: {layer2_inf_t}")
        pseudo_final_suspicious_idxs = np.union1d(suspicious_idxs_2, suspicious_idxs_1).flatten()

        if round >= self.switch_round:
            final_suspicious_idxs = pseudo_final_suspicious_idxs
        print(f"[Combination-result] predicted suspicious set is: {final_suspicious_idxs}")

        # STARTING USING TRUSTWORTHY SCORES
        filtered_suspicious_idxs = list(final_suspicious_idxs.copy())
        if round >= self.switch_round:
            filtered_suspicious_idxs = [idx for idx in final_suspicious_idxs if np.average(self.trustworthy_scores[g_user_indices[idx]]) < self.trustworthy_threshold]
        print(f"[Filtered-result] predicted suspicious set is: {filtered_suspicious_idxs}")          
        if not filtered_suspicious_idxs:
            filtered_suspicious_idxs = suspicious_idxs_1   
        if self.use_trustworthy: # used for ablation study
            final_suspicious_idxs = filtered_suspicious_idxs
        print(f"[Final-result] predicted suspicious set is: {filtered_suspicious_idxs}")   

        for idx, g_idx in enumerate(g_user_indices):
            if idx in final_suspicious_idxs:
                self.trustworthy_scores[g_idx].append(self.lambda_1)
            else:
                self.trustworthy_scores[g_idx].append(self.lambda_2)

        
        # end_fedgrad_t = time.time()*1000
        # fedgrad_t = end_fedgrad_t - start_fedgrad_t # finish calculating the computation time of FedGrad
        
        # tpr_fedgrad, fpr_fedgrad, tnr_fedgrad = 1.0, 1.0, 1.0
        neo_net_list = []
        neo_net_freq = []
        selected_net_indx = []
        for idx, net in enumerate(client_models):
            if idx not in final_suspicious_idxs:
                neo_net_list.append(net)
                neo_net_freq.append(1.0)
                selected_net_indx.append(idx)
        if len(neo_net_list) == 0:
            return [], []
            
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in neo_net_list]
        num_dps = list(num_dps.values())
        
        selected_num_dps = np.asarray(num_dps)[selected_net_indx]
        reconstructed_freq = [snd/sum(selected_num_dps) for snd in selected_num_dps]

        dba.logger.info("Num data points: {}".format(num_dps))
        dba.logger.info("Num selected data points: {}".format(selected_num_dps))
        dba.logger.info("The chosen ones are users: {}, which are global users: {}".format(selected_net_indx, [g_user_indices[ti] for ti in selected_net_indx]))
        
        # aggregated_grad = np.average(vectorize_nets, weights=reconstructed_freq, axis=0).astype(np.float32)

        # aggregated_model = client_models[0] # slicing which doesn't really matter
        # load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device))
        # pred_g_attacker = [g_user_indices[i] for i in final_attacker_idxs]
        # # print(self.pairwise_cs)
        # neo_net_list = [aggregated_model]
        # neo_net_freq = [1.0]
        return selected_net_indx, reconstructed_freq

    def get_compromising_scores(self, global_update, weight_update):
        cs_dist = get_cs_on_base_net(weight_update, global_update)
        score = np.array(cs_dist)
        norm_score = min_max_scale(score)
        return norm_score
