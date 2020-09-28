import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from pytorch_model.utils import prepare_sequence, log_sum_exp, argmax

START_TAG = "<START>"
STOP_TAG = "<STOP>"


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        # LSTM的输出映射到tag空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 定义转移矩阵参数，[i,j]表示从tag_j转移到tag_i
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # 对于转移矩阵，通过初始化值，使得永远不会转移到START_TAG，也永远不会从STOP_TAG开始转移
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        """
        计算所所有路径的score
        :param feats: 发射概率，shape(sentence_size, tagset_size)
        :return:
        """
        # 定义初始状态，START_TAG=0，其他
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas

        # 迭代sentence中的每个token
        for feat in feats:
            alphas_t = []  # 当前位置各路径得分
            # 每个label计算score
            # next_tag的发射概率 + 前一时刻所有tag转移到next_tag的转移概率 + 前一时刻的所有路径得分
            for next_tag in range(self.tagset_size):
                # 对于同一个next_tag，发射概率是相同的，跟前面的tag无关。
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # 转移得分是从所有tag_i转移到next_tag的转移概率
                trans_score = self.transitions[next_tag].view(1, -1)
                # 加上前一时刻的前向变量
                next_tag_var = forward_var + trans_score + emit_score
                # 这个tag的前向变量（forward variable）是所有得分的log-sum-exp
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)  # shape(1, tagset_size), 前一时刻各路径得分，动规中previous
        # 计算最后一步的
        terminal_val = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_val)
        return alpha

    def _get_lstm_features(self, sentence):
        """
        求发射概率
        :param sentence:
        :return:
        """
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)  # shape(sentence_len, 1, embedding_dim)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # shape(sentence_len, 1, hidden_dim)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)  # shape(sentence_len, hidden_dim)
        # 通过一层非线性层，映射到tag空间，得到发射概率
        lstm_feats = self.hidden2tag(lstm_out)  # shape(sentence_len, tagset_size)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        """
        计算给定tag序列的score(计算gold_score)

        :param feats: 发射概率，shape(sentence_size, tagset_size)
        :param tags: sold tags
        :return:
        """
        # sentence序列长度内，依据tag，发射概率与转移概率之和。
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[START_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        """
        使用维特比算法解码
        :param feats:
        :return:
        """
        backpointers = []
        # 初始化,start_tag是0
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0.

        # forward_var 存放上一步的结果变量（viterbi variables）
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # 记录当前时刻抵达next_tag的最佳路径的前一个tag的下标（backpointers ）
            viterbivars_t = []  # 记录当前时刻以next_tag结尾的的结果变量

            for next_tag in range(self.tagset_size):
                # 加上转移到next_tag的概率
                # 计算argmax不需要加上发射概率，因为得到相同next_tag的发射概率都一样的，在最后加上就行
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 加上发射概率
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # 加入STOP_TAG的路径
        terminal_val = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_val)
        path_score = terminal_val[0][best_tag_id]

        # 通过backpointers来解码最优路径,从后往前
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        """
        all_path_score-real_path_score作为loss
        :param sentence:
        :param tags:
        :return:
        """
        # 取lstm层求得的结果
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!
