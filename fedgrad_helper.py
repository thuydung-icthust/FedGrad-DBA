import torch 
import dba

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])

def load_model_weight(net, weight):
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data =  weight[index_bias:index_bias+p.numel()].view(p.size())
        index_bias += p.numel()

def extract_classifier_layer(net_list, global_avg_net, prev_net, model="vgg9"):
    bias_list = []
    weight_list = []
    weight_update = []
    avg_bias = None
    avg_weight = None
    prev_avg_bias = None
    prev_avg_weight = None
    last_model_layer = "classifier" if model=="vgg9" else "fc3" 
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
        # assert (mode in ("krum", "multi-krum"))
        self.num_valid = num_valid
        self.num_workers = num_workers
        self.s = num_adv
        self.instance = instance
        self.choosing_frequencies = {}
        self.accumulate_t_scores = {}
        self.use_trustworthy = use_trustworthy
        self.pairwise_w = np.zeros((total_workers+1, total_workers+1))
        self.pairwise_b = np.zeros((total_workers+1, total_workers+1))
        
        logger.info("Starting performing KrMLRFL...")
        self.pairwise_choosing_frequencies = np.zeros((total_workers, total_workers))
        self.trustworthy_scores = [[0.5] for _ in range(total_workers+1)]


    def exec(self, client_models, num_dps, net_freq, net_avg, g_user_indices, pseudo_avg_net, round, selected_attackers, model_name, device, *args, **kwargs):
        from sklearn.cluster import KMeans
        start_fedgrad_t = time.time()*1000
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        trusted_models = []
        neighbor_distances = []
        logger.info("Starting performing KrMLRFL...")
        # FOR LAYER 1 ONLY
        layer1_start_t = time.time()*1000
        bias_list, weight_list, avg_bias, avg_weight, weight_update, glob_update, prev_avg_weight = extract_classifier_layer(client_models, pseudo_avg_net, net_avg, model_name)
        
        total_client = len(g_user_indices)
        
        raw_t_score = self.get_trustworthy_scores(glob_update, weight_update)
        t_score = []
        for idx, cli in enumerate(g_user_indices):
            # increase the frequency of the selected choosen clients
            self.choosing_frequencies[cli] = self.choosing_frequencies.get(cli, 0) + 1
            # update the accumulator
            self.accumulate_t_scores[cli] = ((self.choosing_frequencies[cli] - 1) / self.choosing_frequencies[cli]) * self.accumulate_t_scores.get(cli, 0) + (1 / self.choosing_frequencies[cli]) *  raw_t_score[idx]
            t_score.append(self.accumulate_t_scores[cli])
        
        t_score = np.array(t_score)
        threshold = min(0.5, np.median(t_score))
        
        participated_attackers = []
        for in_, id_ in enumerate(g_user_indices):
            if id_ in selected_attackers:
                participated_attackers.append(in_)
        
        attacker_local_idxs = [ind_ for ind_ in range(len(g_user_indices)) if t_score[ind_] > threshold]
        print("[T_SCORE] attacker_local_idxs is: ", attacker_local_idxs)
        layer1_end_t = time.time()*1000
        layer1_inf_time = layer1_end_t-layer1_start_t
        print(f"layer1_inf_time: {layer1_inf_time}")
        
        #STARTING THE SECOND LAYER
        layer2_start_t = time.time()*1000
        round_bias_pairwise = np.zeros((total_client, total_client))
        round_weight_pairwise = np.zeros((total_client, total_client))
        
        sum_diff_by_label, glob_temp_sum_by_label = calculate_sum_grad_diff(meta_data = weight_update, num_w = weight_update[0].shape[-1], glob_update=glob_update)
        norm_bias_list = normalize(bias_list, axis=1)
        norm_grad_diff_list = normalize(sum_diff_by_label, axis=1)
        
        # UPDATE CUMULATIVE COSINE SIMILARITY 
        for i, g_i in enumerate(g_user_indices):
            distance = []
            for j, g_j in enumerate(g_user_indices):
                # if i != j:
                
                self.pairwise_choosing_frequencies[g_i][g_j] = self.pairwise_choosing_frequencies[g_i][g_j] + 1.0
                bias_p_i = norm_bias_list[i]
                bias_p_j = norm_bias_list[j]
                cs_1 = np.dot(bias_p_i, bias_p_j)/(np.linalg.norm(bias_p_i)*np.linalg.norm(bias_p_j))
                round_bias_pairwise[i][j] = cs_1.flatten()
                
                w_p_i = norm_grad_diff_list[i]
                w_p_j = norm_grad_diff_list[j]
                cs_2 = np.dot(w_p_i, w_p_j)/(np.linalg.norm(w_p_i)*np.linalg.norm(w_p_j))
                round_weight_pairwise[i][j] = cs_2.flatten()
       
        ## compute scores by KRUM*
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i+1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(float(np.linalg.norm(g_i-g_j)**2)) # let's change this to pytorch version
            neighbor_distances.append(distance)
        # compute scores
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
        i_star = scores.index(min(scores))
        trusted_index = i_star # used for get labels of attackers
        
        
        # # use krum as the baseline to improve, mark the one chosen by krum as trusted
        # if self.num_valid == 1:
        #     i_star = scores.index(min(scores))
        #     logger.info("@@@@ The chosen trusted worker is user: {}, which is global user: {}".format(scores.index(min(scores)), g_user_indices[scores.index(min(scores))]))
        #     trusted_models.append(i_star)
        # else:
        #     topk_ind = np.argpartition(scores, nb_in_score+2)[:self.num_valid]
            
        #     # we reconstruct the weighted averaging here:
        #     selected_num_dps = np.array(num_dps)[topk_ind]
        #     logger.info("Num selected data points: {}".format(selected_num_dps))
        #     logger.info("The chosen ones are users: {}, which are global users: {}".format(topk_ind, [g_user_indices[ti] for ti in topk_ind]))

        #     for ind in topk_ind:
        #         trusted_models.append(ind)
        

        scaler = MinMaxScaler()
        round_bias_pairwise = scaler.fit_transform(round_bias_pairwise)
        round_weight_pairwise = scaler.fit_transform(round_weight_pairwise)

        for i, g_i in enumerate(g_user_indices):
            for j, g_j in enumerate(g_user_indices):
                freq_appear = self.pairwise_choosing_frequencies[g_i][g_j]
                self.pairwise_w[g_i][g_j] = (freq_appear - 1)/freq_appear*self.pairwise_w[g_i][g_j] +  1/freq_appear*round_weight_pairwise[i][j]
                self.pairwise_b[g_i][g_j] = (freq_appear - 1)/freq_appear*self.pairwise_b[g_i][g_j] +  1/freq_appear*round_bias_pairwise[i][j]
                
        
        # From now on, trusted_models contain the index base models treated as valid users.
        attacker_local_idxs_2 = []
        saved_pairwise_sim = []
        layer2_inf_t = 0.0

        # np_scores = np.asarray(scores)
        final_attacker_idxs = attacker_local_idxs # for the first filter
        # NOW CHECK FOR ROUND 50
        if round >= 1: 
            # TODO: find dynamic threshold
            
            cummulative_w = self.pairwise_w[np.ix_(g_user_indices, g_user_indices)]
            cummulative_b = self.pairwise_b[np.ix_(g_user_indices, g_user_indices)]
            
            
            saved_pairwise_sim = np.hstack((cummulative_w, cummulative_b))
            kmeans = KMeans(n_clusters = 2)

            pred_labels = kmeans.fit_predict(saved_pairwise_sim)
            # centroids = kmeans.cluster_centers_
            # np_centroids = np.asarray(centroids)
            
            
            # cls_0_idxs = np.argwhere(np.asarray(pred_labels) == 0).flatten()
            # cls_1_idxs = np.argwhere(np.asarray(pred_labels) == 1).flatten()
            # dist_0 = np.sqrt(np.sum(np.square(saved_pairwise_sim[cls_0_idxs]-np_centroids[0])))/len(cls_0_idxs)
            # dist_1 = np.sqrt(np.sum(np.square(saved_pairwise_sim[cls_1_idxs]-np_centroids[1])))/len(cls_1_idxs)
            # print(f"dist_0 is {dist_0}, dist_1 is {dist_1}")
            
            
            trusted_label = pred_labels[trusted_index]
            label_attack = 0 if trusted_label == 1 else 1
        
            pred_attackers_indx_2 = np.argwhere(np.asarray(pred_labels) == label_attack).flatten()
            print("[PAIRWISE] pred_attackers_indx: ", pred_attackers_indx_2)
            # pred_normal_client = [_id for _id in range(total_client) if _id not in pred_attackers_indx_2]

            #FOR LOGGING ONLY
            # adv_krum_s = np_scores[pred_attackers_indx_2]
            # ben_krum_s = np_scores[pred_normal_client]
            # adv_krum_s_avg = np.average(adv_krum_s).flatten()[0]
            # ben_krum_s_avg = np.average(ben_krum_s).flatten()[0]
            # print(f"trusted client score is: {np_scores[i_star]}")
            # print(f"attackers' scores by krum are: {np_scores[pred_attackers_indx_2]}")
            # print(f"adv_krum_s_avg: {adv_krum_s_avg}")
            # print(f"pred_normal_client's score by krum are: {np_scores[pred_normal_client]}")
            # print(f"ben_krum_s_avg: {ben_krum_s_avg}")

            # missed_attacker_idxs_by_kmeans = [at_id for at_id in participated_attackers if at_id not in pred_attackers_indx_2]

            attacker_local_idxs_2 = pred_attackers_indx_2
            layer2_end_t = time.time()*1000
            layer2_inf_t = layer2_end_t-layer2_start_t
            print(f"layer2_inf_t: {layer2_inf_t}")
            pseudo_final_attacker_idxs = np.union1d(attacker_local_idxs_2, attacker_local_idxs).flatten()

            if round >= 50:
                final_attacker_idxs = pseudo_final_attacker_idxs
            print("assumed final_attacker_idxs: ", pseudo_final_attacker_idxs)
            print(f"final_attacker_idxs is: {final_attacker_idxs}")

        # STARTING USING TRUSTWORTHY SCORES
        normal_idxs = [id_ for id_ in range(total_client) if id_ not in final_attacker_idxs]
        g_attacker_idxs = g_user_indices[final_attacker_idxs]
        print(f"g_attacker_idxs: {g_attacker_idxs}")
        g_normal_idxs = g_user_indices[normal_idxs]
        print(f"g_normal_idxs: {g_normal_idxs}")
        g_attacker_scores = [np.average(self.trustworthy_scores[id_]) for id_ in g_attacker_idxs]
        g_normal_scores = [np.average(self.trustworthy_scores[id_]) for id_ in g_normal_idxs]
        print(f"g_attacker_idxs score: {g_attacker_scores}")
        print(f"g_normal_idxs score: {g_normal_scores}")
        
        trustworthy_threshold = 0.75 #TODO
        filtered_attacker_idxs = list(final_attacker_idxs.copy())
        if round >= 50:
            for idx in final_attacker_idxs:
                g_idx = g_user_indices[idx]
                if np.average(self.trustworthy_scores[g_idx]) >= trustworthy_threshold:
                    filtered_attacker_idxs.remove(idx)
        if not filtered_attacker_idxs:
            filtered_attacker_idxs = attacker_local_idxs
        print(f"filtered_attacker_idxs: {filtered_attacker_idxs}")   
        if self.use_trustworthy:
            final_attacker_idxs = filtered_attacker_idxs
        print(f"final_attacker_idxs are: {final_attacker_idxs}")
        
        for idx, g_idx in enumerate(g_user_indices):
            if idx in final_attacker_idxs:
                self.trustworthy_scores[g_idx].append(0.25)
            else:
                self.trustworthy_scores[g_idx].append(1.0)
        
        #GET ADDITIONAL INFORMATION of TPR and FPR, TNR
        # tp_fedgrad_pred = []
        # for id_ in participated_attackers:
        #     tp_fedgrad_pred.append(1.0 if id_ in final_attacker_idxs else 0.0)
        # fp_fegrad = len(final_attacker_idxs) - sum(tp_fedgrad_pred)
        
        # Calculate true positive rate (TPR = TP/(TP+FN))
        # total_positive = len(participated_attackers)
        # total_negative = total_client - total_positive
        # tpr_fedgrad = 1.0
        # if total_positive > 0.0:
        #     tpr_fedgrad = sum(tp_fedgrad_pred)/total_positive
        # # False postive rate
        # fpr_fedgrad = fp_fegrad/total_negative
        # tnr_fedgrad = 1.0 - fpr_fedgrad
             
        # freq_participated_attackers = [self.choosing_frequencies[g_idx] for g_idx in g_user_indices]
        
        end_fedgrad_t = time.time()*1000
        fedgrad_t = end_fedgrad_t - start_fedgrad_t
        
        
        tpr_fedgrad, fpr_fedgrad, tnr_fedgrad = 1.0, 1.0, 1.0
        neo_net_list = []
        neo_net_freq = []
        selected_net_indx = []
        for idx, net in enumerate(client_models):
            if idx not in final_attacker_idxs:
                neo_net_list.append(net)
                neo_net_freq.append(1.0)
                selected_net_indx.append(idx)
        if len(neo_net_list) == 0:
            neo_net_list.append(client_models[i_star])
            selected_net_indx.append(i_star)
            pred_g_attacker = [g_user_indices[i] for i in final_attacker_idxs]
            # return [client_models[i_star]], [1.0], pred_g_attacker
            return [net_avg], [1.0], [], tpr_fedgrad, fpr_fedgrad, tnr_fedgrad, layer1_inf_time, layer2_inf_t, fedgrad_t
            
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in neo_net_list]
        selected_num_dps = np.array(num_dps)[selected_net_indx]
        reconstructed_freq = [snd/sum(selected_num_dps) for snd in selected_num_dps]

        logger.info("Num data points: {}".format(num_dps))
        logger.info("Num selected data points: {}".format(selected_num_dps))
        logger.info("The chosen ones are users: {}, which are global users: {}".format(selected_net_indx, [g_user_indices[ti] for ti in selected_net_indx]))
        
        aggregated_grad = np.average(vectorize_nets, weights=reconstructed_freq, axis=0).astype(np.float32)

        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device))
        pred_g_attacker = [g_user_indices[i] for i in final_attacker_idxs]
        # print(self.pairwise_cs)
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq, pred_g_attacker, tpr_fedgrad, fpr_fedgrad, tnr_fedgrad, layer1_inf_time, layer2_inf_t, fedgrad_t

    def get_trustworthy_scores(self, global_update, weight_update):
        cs_dist = get_cs_on_base_net(weight_update, global_update)
        score = np.array(cs_dist)
        norm_score = min_max_scale(score)
        
        return norm_score
        # for cli_ind, weight_update in enumerate(weight_update):

    def get_predicted_attackers(self, weight_list, avg_weight, weight_update, total_client):
        # from sklearn.cluster import KMeans
        eucl_dis, cs_dis = get_distance_on_avg_net(weight_list, avg_weight, weight_update, total_client)
        norm_cs_data = min_max_scale(cs_dis)
        norm_eu_data = 1.0 - min_max_scale(eucl_dis)
        # norm_eu_data = min_max_scale(eucl_dis)
        stack_dis = np.hstack((norm_cs_data,norm_eu_data))
        print("stack dis is: ", stack_dis)
        temp_score = [0.5*norm_cs_data[i] + 0.5*norm_eu_data[i] for i in range(total_client)]
        threshold = sum(temp_score)/total_client
        abnormal_score = [1.0 if temp_score[i] > threshold else 0.0 for i in range(total_client)]
        print("abnormal_score: ", abnormal_score)
        
        hb_clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40,
                                metric='euclidean', min_cluster_size=2, min_samples=None, p=None)
        hb_clusterer.fit(stack_dis)
        print("hb_clusterer.labels_ is: ", hb_clusterer.labels_)
        return abnormal_score
