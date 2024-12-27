import torch
import numpy as np
from torch.autograd import Variable
import json
import torch.nn.functional as F
import torch.nn as nn
import ipdb
from collections import Counter

from models.base_model import BaseModel
from modules.kg_reasoning.reasongnn import ReasonGNNLayer
from modules.question_encoding.lstm_encoder import LSTMInstruction
from modules.question_encoding.bert_encoder import BERTInstruction
from modules.layer_init import TypeLayer
from modules.query_update import AttnEncoder, Fusion, QueryReform

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000



class ReaRev(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        Init ReaRev model.
        """
        super(ReaRev, self).__init__(args, num_entity, num_relation, num_word)
        #self.embedding_def()
        #self.share_module_def()
        self.norm_rel = args['norm_rel']
        self.layers(args)
        #self.erase_dict = erase_dict
        

        self.loss_type =  args['loss_type']
        self.num_iter = args['num_iter']
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        self.alg = args['alg']
        assert self.alg == 'bfs'
        self.lm = args['lm']
        
        self.private_module_def(args, num_entity, num_relation)

        self.to(self.device)
        self.lin = nn.Linear(3*self.entity_dim, self.entity_dim)

        self.fusion = Fusion(self.entity_dim)
        self.reforms = []
        for i in range(self.num_ins):
            self.add_module('reform' + str(i), QueryReform(self.entity_dim))
        # self.reform_rel = QueryReform(self.entity_dim)
        # self.add_module('reform', QueryReform(self.entity_dim))

    def layers(self, args):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim

        #self.lstm_dropout = args['lstm_dropout']
        self.linear_dropout = args['linear_dropout']
        
        self.entity_linear = nn.Linear(in_features=self.ent_dim, out_features=entity_dim)
        self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        #self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)

        # dropout
        #self.lstm_drop = nn.Dropout(p=self.lstm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        if self.encode_type:
            self.type_layer = TypeLayer(in_features=entity_dim, out_features=entity_dim,
                                        linear_drop=self.linear_drop, device=self.device, norm_rel=self.norm_rel)

        self.self_att_r = AttnEncoder(self.entity_dim)
        #self.self_att_r_inv = AttnEncoder(self.entity_dim)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        if self.encode_type:
	#Debug Print
            #print("Encode")
            local_entity_emb = self.type_layer(local_entity=local_entity,
                                               edge_list=kb_adj_mat,
                                               rel_features=rel_features)
        else:
		
	#Debug Print
            #print("NotEncode")
            local_entity_emb = self.entity_embedding(local_entity)  # batch_size, max_local_entity, word_dim
            local_entity_emb = self.entity_linear(local_entity_emb)
        
        return local_entity_emb
    
   
    def get_rel_feature(self):
        """
        Encode relation tokens to vectors.
        """
        if self.rel_texts is None:
            rel_features = self.relation_embedding.weight
            rel_features_inv = self.relation_embedding_inv.weight
            rel_features = self.relation_linear(rel_features)
            rel_features_inv = self.relation_linear(rel_features_inv)
        else:
            
            rel_features = self.instruction.question_emb(self.rel_features)
            rel_features_inv = self.instruction.question_emb(self.rel_features_inv)
            
            rel_features = self.self_att_r(rel_features,  (self.rel_texts != self.instruction.pad_val).float())
            rel_features_inv = self.self_att_r(rel_features_inv,  (self.rel_texts != self.instruction.pad_val).float())
            if self.lm == 'lstm':
                rel_features = self.self_att_r(rel_features, (self.rel_texts != self.num_relation+1).float())
                rel_features_inv = self.self_att_r(rel_features_inv, (self.rel_texts_inv != self.num_relation+1).float())

        return rel_features, rel_features_inv


    def private_module_def(self, args, num_entity, num_relation):
        """
        Building modules: LM encoder, GNN, etc.
        """
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim
        self.reasoning = ReasonGNNLayer(args, num_entity, num_relation, entity_dim, self.alg)
        if args['lm'] == 'lstm':
            self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)
            self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        else:
            self.instruction = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'])
            #self.relation_linear = nn.Linear(in_features=self.instruction.word_dim, out_features=entity_dim)
        # self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=entity_dim, out_features=entity_dim)

    #EE Edit Build Fact Matrix
    def _build_fact_mat(self, kb_adj_mat, P, threshold):
        """
        Creates local adj mats that contain entities, relations, and structure.
        """
        with open("edge.json", 'r') as file2:
            erase_dict = json.load(file2)
        batch_heads = kb_adj_mat[0]
        batch_rels = kb_adj_mat[1]
        batch_tails = kb_adj_mat[2]
        batch_ids = kb_adj_mat[3]
        #print(sample_ids)
        change_list = torch.nonzero(P > threshold)
        #ipdb.set_trace()
        add_heads = np.array([], dtype = int)
        add_rels = np.array([], dtype = int)
        add_tails = np.array([], dtype = int)
        add_ids = np.array([], dtype = int)
        
        for (idx, head, tail) in change_list:

            #real_head = head + idx * 2000
            #real_tail = tail + idx * 2000
            real_head = head
            real_tail = tail
            real_head = real_head.item()
            real_tail = real_tail.item()
        
            if (str(idx.item()) in erase_dict) and (str(real_head) in erase_dict[str(idx.item())]) and str(real_tail) in erase_dict[str(idx.item())][str(real_head)]:
                add_heads = np.append(add_heads,real_head + 2000 * idx.item())
		
                add_rels = np.append(add_rels, erase_dict[str(idx.item())][str(real_head)][str(real_tail)])
                add_tails = np.append(add_tails,real_tail + 2000 * idx.item())
                add_ids = np.append(add_ids, idx)
                #ipdb.set_trace()
        batch_heads = np.append(batch_heads, add_heads)
        batch_rels = np.append(batch_rels, add_rels)
        batch_tails = np.append(batch_tails, add_tails)
        batch_ids = np.append(batch_ids, add_ids)
        #ipdb.set_trace()

        fact_ids = np.array(range(len(batch_heads)), dtype=int)
        head_rels_ids = zip(batch_heads, batch_rels)
        head_count = Counter(batch_heads)
        # tail_count = Counter(batch_tails)
        weight_list = [1.0 / head_count[head] for head in batch_heads]

        
        head_rels_batch = list(zip(batch_heads, batch_rels))
        #print(head_rels_batch)
        head_rels_count = Counter(head_rels_batch)
        weight_rel_list = [1.0 / head_rels_count[(h,r)] for (h,r) in head_rels_batch]

        #print(head_rels_count)

        # tail_count = Counter(batch_tails)

        # entity2fact_index = torch.LongTensor([batch_heads, fact_ids])
        # entity2fact_val = torch.FloatTensor(weight_list)
        # entity2fact_mat = torch.sparse.FloatTensor(entity2fact_index, entity2fact_val, torch.Size(
        #     [len(sample_ids) * self.max_local_entity, len(batch_heads)]))
        return batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list, weight_rel_list


    #EE Edit Distance Calculate
    def compute_distance(self, batch_size):

        distances = torch.zeros(20, 2000,2000)
        n_batch, n_entitys, n_features = self.local_entity_emb.shape
        for batch_idx in range(n_batch):
            batch_tensor = self.local_entity_emb[batch_idx]
            squared_norms = (batch_tensor ** 2).sum(dim=1)
            dot_product = torch.matmul(batch_tensor, batch_tensor.T)
            pairwise_distances_squared = squared_norms.unsqueeze(0) + squared_norms.unsqueeze(1) - 2 * dot_product
            pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)  # WYQ: Prevent negative values
            pairwise_distances = torch.sqrt(pairwise_distances_squared) # + 1e-8 Add small epsilon
            softmax_distances = F.softmax(-pairwise_distances, dim = 1)
		
            distances[batch_idx] = softmax_distances



        return distances

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities):
        """
        Initializing Reasoning
        """
        self.local_entity = local_entity
        batch_size = local_entity.size(0)
        self.instruction_list, self.attn_list = self.instruction(q_input)
        rel_features, rel_features_inv  = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, rel_features)
        self.init_entity_emb = self.local_entity_emb
        self.curr_dist = curr_dist
        self.dist_history = []
        self.action_probs = []
        self.seed_entities = curr_dist
	#EE Edit
        # import ipdb; ipdb.set_trace()
        #print(type(self.local_entity_emb))
        #print(self.local_entity_emb.shape) #[20, 2000. 50] 20 2000 50
        #print(self.local_entity_emb.size())
	

            #if i < 10:
                #print(f"head{head}, rel{rel}, tail{tail}, bid{bid}")
        #print(type(Score))
        #print(Score.shape) #[20, 2000, 2000]
        #print(Score.size())
       #for item in kb_adj_mat[1]:	
       #    if item == 0:
       #        print("0rel")        
	#print(kb_adj_mat[0])

        #print(type(Sim))
        #print(Sim.shape) #[20, 2000, 2000]
        #print(Sim.size())
	#P = SIM X S^T Elementwise Product (1 - S)

        #print(P[0, 0:5, 0:5])
        #ipdb.set_trace()
	#print(type(P))
        #print(P.shape) #[20, 2000, 2000]
        #print(P.size())

        Score = torch.zeros(20, 2000,2000)
        for i, (head,rel,tail, bid) in enumerate(zip(kb_adj_mat[0],kb_adj_mat[1],kb_adj_mat[2],kb_adj_mat[3])):
            Score[bid][head - bid * 2000][tail - bid * 2000] = 1
        Sim = self.compute_distance(batch_size)
        P = torch.bmm(Sim, Score.transpose(1,2)) * (1 - Score)
        threshold = 0.9
        kb_adj_mat = self._build_fact_mat(kb_adj_mat, P, threshold)
        #self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, rel_features)

        #ipdb.set_trace()
	
        self.reasoning.init_reason( 
                                   local_entity=local_entity,
                                   kb_adj_mat=kb_adj_mat,
                                   local_entity_emb=self.local_entity_emb,
                                   rel_features=rel_features,
                                   rel_features_inv=rel_features_inv,
                                   query_entities=query_entities)


    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    
    def forward(self, batch, training=False):
        """
        Forward function: creates instructions and performs GNN reasoning.
        """

        # local_entity, query_entities, kb_adj_mat, query_text, seed_dist, answer_dist = batch
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        # local_entity_mask = (local_entity != self.num_entity).float()
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        q_input= torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        #query_text2 = torch.from_numpy(query_text2).type('torch.LongTensor').to(self.device)
        if self.lm != 'lstm':
            pad_val = self.instruction.pad_val #tokenizer.convert_tokens_to_ids(self.instruction.tokenizer.pad_token)
            query_mask = (q_input != pad_val).float()
            
        else:
            query_mask = (q_input != self.num_word).float()

        
        """
        Instruction generations
        """
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)
        self.instruction.init_reason(q_input)
        for i in range(self.num_ins):
            relational_ins, attn_weight = self.instruction.get_instruction(self.instruction.relational_ins, step=i) 
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins
        #relation_ins = torch.cat(self.instruction.instructions, dim=1)
        #query_emb = None
        self.dist_history.append(self.curr_dist)


        """
        BFS + GNN reasoning
        """

        for t in range(self.num_iter):
            relation_ins = torch.cat(self.instruction.instructions, dim=1)
            self.curr_dist = current_dist            
            for j in range(self.num_gnn):
                self.curr_dist, global_rep = self.reasoning(self.curr_dist, relation_ins, step=j)
            self.dist_history.append(self.curr_dist)
            qs = []

            """
            Instruction Updates
            """
            for j in range(self.num_ins):
                reform = getattr(self, 'reform' + str(j))
                q = reform(self.instruction.instructions[j].squeeze(1), global_rep, query_entities, local_entity)
                qs.append(q.unsqueeze(1))
                self.instruction.instructions[j] = q.unsqueeze(1)
        
        
        """
        Answer Predictions
        """
        pred_dist = self.dist_history[-1]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        # filter no answer training case
        # loss = 0
        # for pred_dist in self.dist_history:
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)

        
        pred_dist = self.dist_history[-1]
        pred = torch.max(pred_dist, dim=1)[1]
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list

    
