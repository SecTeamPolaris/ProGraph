#!/usr/bin/env python
# coding: utf-8

# In[1]:


class data(object):
    def __init__(self):
        self.IP = []
        self.cert = []
        self.urls = []
        self.static_features = []
        self.seq = []
        self.label = []
        self.time = []
        self.image = []
        self.fn = []
        
    def insert(self,IP,cert,urls,static_features,seq,label,time,image,fn):
        self.IP.append(IP)
        self.cert.append(cert)
        self.urls.append(urls)
        self.static_features.append(static_features)
        self.seq.append(seq)
        self.label.append(label)
        self.time.append(time)
        self.image.append(image)
        self.fn.append(fn)


# In[2]:


from interval import Interval
import os,shutil
import numpy as np
import pickle


class cert_cluster(object):
    def __init__(self):
        self.IPs = []
        self.urls = {} #{urls:number}
        self.url_number = [] # sorted (url:number)
        self.url_weight = {}#{url:weight}
        self.url_unique = []
        self.certs = [] 
        self.sessions = []
        self.images = []
        self.static_features = []
        self.fn = []
        

        self.seq = []
        self.label = -1
        self.tag = -1
        self.time_slides = [] #(start,end)
        
        self.sim_list = []
        self.url_all = []
        
        self.ip_weight = {}
        self.cert_weight = {}
        
    #file name
    def fn_cal(self):
        self.fn = [ss['fn'] for ss in self.sessions]
            
        
    #all urls
    def all_url_cal(self):
        for session in self.sessions:
            urls = session['urls']
            for url in urls:
                if url not in self.url_all:
                    self.url_all.append(url)
    #url sim cal
    def sim_cal(self,urls_dict):
        self.sim_list = []
        for url_dict_per in urls_dict:
            score = 0
            for url in self.url_unique:
                if url in url_dict_per:
                    score += 1
            self.sim_list.append(score / (len(self.url_unique)*1.0))
        if np.array(self.sim_list).max() > 0:
            self.sim_list = self.sim_list/np.array(self.sim_list).max()
        max_idx = np.argmax(np.array(self.sim_list))
        for i in range(len(self.sim_list)):
            if i!=max_idx:
                self.sim_list[i] = 0
            
    #ip sim cal
    def sim_ip_cal(self,urls_dict):
        self.sim_list = []
        for url_dict_per in urls_dict:
            score = 0
            for url in self.IPs:
                if url in url_dict_per:
                    score += 1
            self.sim_list.append(score)
        if np.array(self.sim_list).max() > 0:
            self.sim_list = self.sim_list/np.array(self.sim_list).max()
        max_idx = np.argmax(np.array(self.sim_list))
        for i in range(len(self.sim_list)):
            if i!=max_idx:
                self.sim_list[i] = 0
        
            
    
    #static
    def static_cal(self):
        self.static_features = [a['static_features'] for a in self.sessions]
    
    
    #major voting for cluster's label
    def label_cal(self):
        labels = [a['label'] for a in self.sessions]
        self.label = max(labels,key = labels.count)
        
    def tag_cal(self):
        tags = [a['tag'] for a in self.sessions]
        self.tag = max(tags,key = tags.count)
    
    #image extract
    def image_cal(self):
        self.images = [item['image'].flatten() for item in self.sessions]
    
    #seq_ipdaate
    def seq_cal(self):
#         self.seq = [item['seq'].flatten() for item in self.sessions]
        for i in range(len(self.sessions)):
            session = self.sessions[i]
            mat = session['seq']
            sta = session['static_features']
            for item in sta:
                mat = np.append(mat,float(item))

            self.seq.append(mat)
            
            
        
    #time_slieds generation
    def time_cal(self):
        def getfirst(item):
            return item[0]
        for session in self.sessions:
            self.time_slides.append(session['time'])
#         print(self.time_slides)
        self.time_slides = sorted(self.time_slides,key=getfirst)
        
        #concat time period
        self.time_slides = [Interval(item[0],item[1],lower_closed=True, upper_closed=True) for item in self.time_slides]
#         print(len(self.time_slides))
        while True:
            big_flag = 0
            for i in range(len(self.time_slides)):
                
                flag = 0
                for j in range(len(self.time_slides)):
#                     print((i,j))
                    if i ==len(self.time_slides)-1 and j == len(self.time_slides)-1:
                        big_flag =1
                    if i==j:
                        continue      
                    if self.time_slides[i].overlaps(self.time_slides[j]):
                        interval_merge = self.time_slides[i].join(self.time_slides[j])
                        self.time_slides[i] = interval_merge
                        del self.time_slides[j]

                        flag = 1
                        break                
                if flag == 1:
                    break
            if big_flag==1:
                break
        self.time_slides = [(item.lower_bound,item.upper_bound) for item in self.time_slides]
#         print(self.time_slides)
        
                    
                    
                
    def urls_cal(self):
        for item in self.sessions:
            urls = item['urls']
            for url in urls:
                if url not in list(self.urls.keys()):
                    self.urls[url] = 1
                else:
                    self.urls[url] += 1
        #sorted
        self.url_number = sorted(self.urls.items(),key=lambda item:item[1],reverse=True)
    

    def urls_weight(self):
        total = np.array([item[1] for item in self.url_number]).sum()*1.0
        for item in self.url_number:
            self.url_weight[item[0]] = item[1]*1.0/total
                
#         self.url_weight = [(item[0],item[1]*1.0/total) for item in self.url_number]
        self.url_unique = [item[0] for item in self.url_number]
        

    def ips_cal(self):
        for item in self.sessions:
            if item['IP'] not in self.IPs:
                self.IPs.append(item['IP'])
    

    def cert_cal(self):
        for item in self.sessions:
            if item['cert'] not in self.certs:
                self.certs.append(item['cert'])
        self.certs = [item for item in self.certs if item is not 0]
    

    def url_clean(self):
        if len(self.url_number) <=5:
            return
        self.url_number = self.url_number[:5]
        self.url_unique = [item[0] for item in self.url_number]
        self.url_weight = {}
        self.urls_weight()
        
    #ip_weigth
    def ip_weight_cal(self):
        self.ip_weight = {}
        for session in self.sessions:
            if session['IP'] not in list(self.ip_weight.keys()):
                self.ip_weight[session['IP']] = 1.0/(len(self.sessions)*1.0)
            else:
                self.ip_weight[session['IP']] += 1.0/(len(self.sessions)*1.0)

        
        
        
    #cert_weigth
    def cert_weight_cal(self):
        self.cert_weight = {}
        count = 0
        for session in self.sessions:
            if session['cert'] != 0 and session['cert'] not in list(self.cert_weight.keys()):
                self.cert_weight[session['cert']] = 1
                count +=1
            elif session['cert'] != 0 and session['cert'] in list(self.cert_weight.keys()):
                self.cert_weight[session['cert']] += 1
                count +=1
        if count != 0:
            for key in list(self.cert_weight.keys()):
                self.cert_weight[key] = self.cert_weight[key]/(count*1.0)
    
    
    #更新cert_clt
    def update(self):
        self.urls_cal()
        self.urls_weight()
        self.ips_cal()
        self.cert_cal()
        self.seq_cal()
        self.image_cal()
        self.label_cal()
        self.tag_cal()
        self.static_cal()
        self.url_clean()
        self.all_url_cal()
        self.fn_cal()
        
        self.cert_weight_cal()
        self.ip_weight_cal()
        
#         self.time_cal()
                




# In[3]:


def adj_matrix_gen(clusters):
    def cert_sim(clt1,clt2):
        if len(list(clt1.cert_weight.keys())) == 0 or len(list(clt2.cert_weight.keys())) == 0:
            return 0
        cross_keys = list(set(list(clt1.cert_weight.keys()))&set(list(clt2.cert_weight.keys())))
        score = 0.0
        for key in cross_keys:
            score += clt1.cert_weight[key]*clt2.cert_weight[key]
        return score
    def ip_sim(clt1,clt2):
        if len(list(clt1.ip_weight.keys())) == 0 or len(list(clt2.ip_weight.keys())) == 0:
            return 0
        cross_keys = list(set(list(clt1.ip_weight.keys()))&set(list(clt2.ip_weight.keys())))
        score = 0.0
        for key in cross_keys:
            score += clt1.ip_weight[key]*clt2.ip_weight[key]
        return score
    
    def url_sim(clt1,clt2):
        if len(clt1.url_unique) == 0 or len(clt2.url_unique) == 0:
            return 0
        
        overlaps = list(set(clt1.url_unique)&set(clt2.url_unique))
        if len(overlaps) is 0:
            return 0
        res =  np.array([clt1.url_weight[key]*clt2.url_weight[key] for key in overlaps]).sum() / len(overlaps)
#         print(res)
        if res <0.3:
            res = 0
        return res

    
    def time_sim(clt1,clt2):
        clt1_time = [Interval(item[0],item[1],lower_closed=True, upper_closed=True) for item in clt1.time_slides]
        clt2_time = [Interval(item[0],item[1],lower_closed=True, upper_closed=True) for item in clt2.time_slides]
        count = 0
        for clt1_t in clt1_time:
            for clt2_t in clt2_time:
                if clt1_t.overlaps(clt2_t):
                    count+=1
        if count <=7:
            count = 0
        return count
    
    mat = np.zeros((len(clusters),len(clusters)))
    for i in range(len(clusters)):
        for j in range(i,len(clusters)):
            if i==j:
                continue
            mat[i][j] = url_sim(clusters[i],clusters[j]) + cert_sim(clusters[i],clusters[j]) + ip_sim(clusters[i],clusters[j]) #+ time_sim(clusters[i],clusters[j])
            if i!=j:
                mat[j][i] = mat[i][j]
    return mat


# In[4]:


def clt_analysis(clts, adj_matrix):
    inter_per_list = []
    max_indx_list = []
    outer_score_list_all = []
    for i in range(adj_matrix.shape[0]):
        if clts[i].tag == 0:
            continue
        inter_score = 0.0
        outer_score = 0.0
#         inter_score_list = [0.0 for i in range(10)]
        outer_score_list = [0.0 for i in range(10)]
        
        for j in range(adj_matrix.shape[0]):
            if i==j or clts[j].tag == 0:
                continue
            if clts[i].label == clts[j].label:
                inter_score += adj_matrix[i][j]
                outer_score_list[int(clts[i].label)] += adj_matrix[i][j]
            else:
                outer_score_list[int(clts[j].label)] += adj_matrix[i][j]
                outer_score += adj_matrix[i][j]
        if (inter_score+outer_score) != 0:
            inter_per_list.append(inter_score/(inter_score+outer_score))
        else:
            inter_per_list.append(-1)
            
#         outer_score_list[clts[i].label] = adj_matrix[i][i]
        max_indx_list.append(np.argmax(np.array(outer_score_list)))
        outer_score_list_all.append(outer_score_list)
    outer_clt = []
    for i in range(len(inter_per_list)):
        if inter_per_list[i] <=0.5 and inter_per_list[i] >= 0:
            outer_clt.append((i,clts[i]))
    max_indx_list = [(outer_score_list_all[i][clts[i].label],outer_score_list_all[i], len(clts[i].sessions)) for i in range(len(max_indx_list)) if max_indx_list[i] != int(clts[i].label) and np.array(max_indx_list[i]).sum()>0]
    return outer_clt,max_indx_list
        
        


# In[5]:


def enhc_iso(clts,labels):
    def time_sim(clt1,clt2):
        clt1_time = [Interval(item[0],item[1],lower_closed=True, upper_closed=True) for item in clt1.time_slides]
        clt2_time = [Interval(item[0],item[1],lower_closed=True, upper_closed=True) for item in clt2.time_slides]
        count = 0
        for clt1_t in clt1_time:
            for clt2_t in clt2_time:
                if clt1_t.overlaps(clt2_t):
                    count+=1
        return count
    
    #deal with the isolated clts only
    zero_idx = [i for i in range(len(labels)) if labels[i] == -1]
    
    for idx in zero_idx:
        #for each clt, we calculate the time sim with other clts
        score_list = [0.0 for x in range(10)]
        for j in range(len(clts)):
            if j == idx:
                continue
            score = time_sim(clts[idx],clts[j])
            score_list[labels[j]] += score
        pred_label = np.argmax(np.array(score_list))
        labels[idx] = pred_label
        
    return labels
        
def check_zero_label(adj_matrix,labels):
    idxs = []
    for i in range(adj_matrix.shape[0]):
        if labels[i] == -1:
            continue
        flag = 1
        count = 0
        for j in range(adj_matrix.shape[0]):
            if i==j:
                continue
            if labels[j] == -1:
                continue
            if labels[i] == labels[j] and adj_matrix[i][j] != 0:
                flag = 0
            elif labels[i] != labels[j] and adj_matrix[i][j] != 0:
                count += adj_matrix[i][j]
        if flag and count != 0:
            idxs.append(i)
    return idxs
            
def zero_adj(zero_idxs,clts, adj_matrix):
    for idx in zero_idxs:
        if clts[idx].tag == 0:
            continue
        cand = clts[idx]
        score_list = [0.0 for i in range(10)]
        for i in range(len(clts)):
            if i==idx:
                continue
            score_list[clts[i].label] += adj_matrix[idx][i]
        prob = np.array(softmax(score_list))
        if np.argmax(prob) != clts[idx].label and prob[np.argmax(prob)] >= 0.9:
            clts[idx].label = np.argmax(prob)
    return clts
        

def check_zero(adj_matrix):
    sum_v = np.sum(adj_matrix,axis=0).reshape((adj_matrix.shape[0],1)) - np.array([adj_matrix[i][i] for i in range(adj_matrix.shape[0])]).reshape((adj_matrix.shape[0],1))
    zero_idx = [i for i in range(sum_v.shape[0]) if int(sum_v[i]) == 0]
    print(len(zero_idx))
    return zero_idx

#adjust the label of outer clts
def clt_adj(clts,o_clts):
    def cert_sim(clt1,clt2):
        if len(clt1.certs) == 0 or len(clt2.certs) == 0:
            return 0
        return len(list(set(clt1.certs)&set(clt2.certs)))
    def ip_sim(clt1,clt2):
        return len(list(set(clt1.IPs)&set(clt2.IPs)))
    for o_idx,o_clt in o_clts:
        clss = [0.0 for i in range(10)]
        for i in range(len(clts)):
            if i==o_idx:
                continue
            sim = cert_sim(o_clt,clts[i]) + ip_sim(o_clt,clts[i])
            clss[clts[i].label] += sim
        if np.array(clss).max() < 2:
            print("{} no need for adjustment".format(o_idx))
            continue
        if np.argmax(np.array(clss)) != o_clt.label:
            print("{}: {} -> {}".format(o_idx, o_clt.label,np.argmax(np.array(clss))))
            clts[o_idx].label = np.argmax(np.array(clss))
                           
    return clts

def softmax(x):
    return np.exp(x - np.max(x))/(np.sum(np.exp(x - np.max(x))))


# In[6]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


def spread(clts,adj_matrix,labels):
    #predict the unlabel clts
    for i in range(len(labels)):
        if labels[i] != -1:
            continue
        score_list = [0.0 for x in range(10)]
        for j in range(len(labels)):
            if i==j: #ignore the self-node
                continue
            if labels[j] == -1: #ignore the unlabel testing nodes
                continue
            score_list[labels[j]] += adj_matrix[i][j]
        if np.array(score_list).sum() == 0:
            continue
        score_list = np.array(softmax(score_list))
        pred_label = np.argmax(score_list)
        #tag the predicted label
        labels[i] = pred_label
#         print(pred_label)
    return labels

def acc_per_round(clts,labels):
    gt_labels = [clt.label for clt in clts]
    acc = accuracy_score(gt_labels,labels)
    pre = recall_score(gt_labels,labels,average="weighted")
    rec = precision_score(gt_labels,labels, average="weighted")
    f1 = f1_score(gt_labels,labels,average="weighted")
    return acc,pre,rec,f1
    
    acc = 0
    al = 0
    for i in range(len(labels)):
        if labels[i]==-1 or clts[i].tag == 1:
            continue
        al += 1
        if labels[i] == clts[i].label:
            acc += 1
    return acc*1.0/(al*1.0)

def aggregation(clts,mat,labels):
    #ip,cert,url weight aggragation
    def agg_weight(clt_c,clt_n,weight):
        #ip agg
        for ip in list(clt_n.ip_weight.keys()):
            if ip not in list(clt_c.ip_weight.keys()):
                clt_c.ip_weight[ip] = clt_n.ip_weight[ip] * weight
            else:
                clt_c.ip_weight[ip] += clt_c.ip_weight[ip] * clt_n.ip_weight[ip] * weight
        #cert agg
        for cert in list(clt_n.cert_weight.keys()):
            if cert not in list(clt_c.cert_weight.keys()):
                clt_c.cert_weight[cert] = clt_n.cert_weight[cert] * weight
            else:
                clt_c.cert_weight[cert] += clt_c.cert_weight[cert] * clt_n.cert_weight[cert] * weight
                
        #url agg
        for url in list(clt_n.url_weight.keys()):
            if url not in list(clt_c.url_weight.keys()):
                clt_c.url_weight[url] = clt_n.url_weight[url] * weight
            else:
                clt_c.url_weight[url] += clt_c.url_weight[url] * clt_n.url_weight[url] * weight
                
        
    
    #aggregate the labeled nodes
    for i in range(len(labels)):
        if labels[i] == -1:
            continue
        #aggragate the labeled nodes from same class
        softmax_mat = []
        for k in range(len(labels)):
            if labels[k]!=-1 and labels[k]==labels[i]:
                softmax_mat.append(mat[i][k])
            else:
                softmax_mat.append(0) 
        for j in range((len(labels))):
            if i==j or labels[i] != labels[j]:
                continue
            #we aggregate the labled nodes from same class(so as to calculate the softmax)
            agg_weight(clts[i],clts[j],softmax(softmax_mat)[j]) 
    #update the edge with the unlabeled nodes
    adj_matrix = adj_matrix_gen(clts)
    return adj_matrix
            
def label_update(clts,labels):
    labels = []
    for clt in clts:
        if clt.tag == 0:
            labels.append(-1)
        else:
            labels.append(clt.label)
    labels = np.array(labels)
    return labels


# In[7]:


def label_mat_init(IP_clusters):
    adj_matrix = adj_matrix_gen(IP_clusters)
    labels = []
    for clt in IP_clusters:
        if clt.tag == 0:
            labels.append(-1)
        else:
            labels.append(clt.label)
    labels = np.array(labels)
    return adj_matrix,labels


# In[8]:


def data_init(mode, test_rate = 0):
    #load traning data from net_A
    with open("./dataset/InitTraffic.pkl",'rb') as file:
        dataT  = pickle.loads(file.read())

    data_list = []

    for i in range(len(dataT.label)):
        tmp = {}
        tmp['IP'] = dataT.static_feature[i][0]
        if len(dataT.cert_number[i]) is 0:
            tmp['cert'] = 0
        else:
            tmp['cert'] = dataT.cert_number[i][0]
        tmp['urls'] = [item for item in dataT.urls[i] if item != '0']
        tmp['static_features'] = np.array([float(item) for item in dataT.static_feature[i][1:-2]])
        tmp['seq'] = dataT.seq_matirx[i]
        tmp['label'] = dataT.label[i]
        tmp['time'] = (float(dataT.static_feature[i][-2]),float(dataT.static_feature[i][-1]))
        tmp['image'] = dataT.image[i]
        tmp['fn'] = dataT.fn[i]
        tmp['tag'] = 1

        data_list.append(tmp)
    print("{} sessions loaded from training set.".format(len(data_list)))

    IP_clusters = []
    used_sessions = []

    #inital IP clusters
    for i in range(len(data_list)):
        session = data_list[i]
        IP = session['IP']
        if len(IP_clusters) == 0:
            used_sessions.append(i)
            clt = cert_cluster()
            clt.sessions.append(session)
            clt.update()
            IP_clusters.append(clt)
        else:
            d = 0
            for j in range(len(IP_clusters)):
                if IP in IP_clusters[j].IPs:
                    IP_clusters[j].sessions.append(session)
                    IP_clusters[j].update()
                    used_sessions.append(i)
                    d = 1
                    break

            if d == 0:
                used_sessions.append(i)
                clt = cert_cluster()
                clt.sessions.append(session)
                clt.update()
                IP_clusters.append(clt)

    data_list = [data_list[i] for i in range(len(data_list)) if i not in used_sessions]
    for item in IP_clusters:
        item.time_cal()
    if mode == 'cross':
        print("{} clusters are initialized for training set.".format(len(IP_clusters)))
        
    if mode == 'cross':
        with open("./dataset/TestTraffic.pkl",'rb') as file:
            dataT  = pickle.loads(file.read())

        for i in range(len(dataT.label)):
            tmp = {}
            tmp['IP'] = dataT.static_feature[i][0]
            if len(dataT.cert_number[i]) is 0:
                tmp['cert'] = 0
            else:
                tmp['cert'] = dataT.cert_number[i][0]
            tmp['urls'] = [item for item in dataT.urls[i] if item != '0']
            tmp['static_features'] = np.array([float(item) for item in dataT.static_feature[i][1:-2]])
            tmp['seq'] = dataT.seq_matirx[i]
            tmp['label'] = dataT.label[i]
            tmp['time'] = (float(dataT.static_feature[i][-2]),float(dataT.static_feature[i][-1]))
            tmp['image'] = dataT.image[i]
            tmp['fn'] = dataT.fn[i]
            tmp['tag'] = 0

            data_list.append(tmp)
        print("{} sessions loaded from testing set.".format(len(data_list)))

        IP_clusters_poor = []
        used_sessions = []

        #inital IP clusters
        for i in range(len(data_list)):
            session = data_list[i]
            IP = session['IP']
            if len(IP_clusters_poor) == 0:
                used_sessions.append(i)
                clt = cert_cluster()
                clt.sessions.append(session)
                clt.update()
                IP_clusters_poor.append(clt)
            else:
                d = 0
                for j in range(len(IP_clusters_poor)):
                    if IP in IP_clusters_poor[j].IPs:
                        IP_clusters_poor[j].sessions.append(session)
                        IP_clusters_poor[j].update()
                        used_sessions.append(i)
                        d = 1
                        break

                if d == 0:
                    used_sessions.append(i)
                    clt = cert_cluster()
                    clt.sessions.append(session)
                    clt.update()
                    IP_clusters_poor.append(clt)

        data_list = [data_list[i] for i in range(len(data_list)) if i not in used_sessions]
        for item in IP_clusters_poor:
            item.time_cal()
        IP_clusters.extend(IP_clusters_poor)
        
        print("{} clusters are initialized for testing set.".format(len(IP_clusters_poor)))
        print("{} nodes are included in the initialized graph.".format(len(IP_clusters)))
        return IP_clusters
    else:
        #construct testing set
        label_idx_dict = {}
        for i in range(len(IP_clusters)):
            clt = IP_clusters[i]
            if clt.label not in list(label_idx_dict.keys()):
                label_idx_dict[clt.label] = [i]
            else:
                label_idx_dict[clt.label].append(i)
        test_idx = []
        for key in list(label_idx_dict.keys()):
            idxs = np.arange(len(label_idx_dict[key]))
            np.random.shuffle(idxs)
            idxs = idxs[:int(idxs.shape[0]*test_rate)]
            test_idx.extend(list(idxs))
        for idx in test_idx:
            IP_clusters[idx].tag = 0
        print("{} clusters are initialized for training set.".format(len(IP_clusters) - len(test_idx)))
        print("{} clusters are initialized for testing set.".format(len(test_idx)))
        print("{} nodes are included in the initialized graph.".format(len(IP_clusters)))
        return IP_clusters


# In[9]:


#initialize training set and testing set ->IP_clusters
IP_clusters = data_init(mode='cross', test_rate= 0.5)#ratio of the sessions in test set
#adj_matrix and labels initialization
adj_matrix,labels = label_mat_init(IP_clusters)
#check out outer clusters
outer_clt,max_indx_list = clt_analysis(IP_clusters,adj_matrix)


# In[10]:


#update labels
labels = label_update(IP_clusters,labels)
#check out isolate clusters
zero_idx = check_zero_label(adj_matrix,labels)
zero_clts = [(i, IP_clusters[i]) for i in range(len(IP_clusters)) if i in zero_idx]
adj_cand_clts = outer_clt + zero_clts
#adjust iso clts to the gt classes
print("*** Confused nodes should be adjusted to the ground true labels.***")
clts_adj = clt_adj(IP_clusters,adj_cand_clts)

#start to propagate
print()
print("Starting to propagate.")
epoch = 3 # A small epoch can achieve satisfactory performance and prevent from overfitting
for e in range(epoch):
    # nodes aggregate info from neighbours and update the adjoin edges
    adj_matrix = aggregation(clts_adj,adj_matrix,labels)
    # update the labels
    labels = label_update(clts_adj,labels)
    # spread label info to the unlabeled neighbours
    labels = spread(clts_adj,adj_matrix,labels)
    # performing measurement in testing set
    acc,pre,rec,f1 = acc_per_round(clts_adj,labels)
    f1 = 2*pre*rec/(pre+rec)
    print("Epoch : {}  Acc: {} pre: {} rec: {} f1: {}.".format(e,acc,pre,rec,f1))
# deal with the isolated nodes, aggregate by time info
labels = enhc_iso(clts_adj,labels)
f1 = 2*pre*rec/(pre+rec)
acc,pre,rec,f1 = acc_per_round(clts_adj,labels)
# acc after propagation
print("Testing Acc: {} pre: {} rec: {} f1: {} after {} epochs.".format(acc,pre,rec,f1,epoch))



# In[ ]:




