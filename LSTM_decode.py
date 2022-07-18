import torch
import numpy as np
import torch.nn as nn
from LSTM.LSTM_model import M_LSTM
from BPE_handle.word22id import interchange, word2id, id2word, get_word, replace_word

PATH_MODEL = "./LSTM_train_state/lstm_train_state_73000.pth"
PATH_INDEX = "F:/PyTorch学习/BPE_handle/en_index.npy"
# PATH_INDEX = "./LSTM_file/en_index.npy"

num_layers = 6


def load(srcf):
    v = np.load(srcf, allow_pickle=True).item()  # 字典的导入
    return v


if __name__ == '__main__':
    en2index = load(PATH_INDEX)
    en = get_word(en2index)

    sentence = input("请输入测试语句:").strip().split()
    # print(sentence)

    sentence = replace_word(sentence, en)
    # print(sentence)

    word2id_l = word2id(en2index, sentence)
    # print(word2id_l)

    model = M_LSTM(len(en2index), num_layers)
    net_state_dict = torch.load(PATH_MODEL, map_location='cpu')
    model.load_state_dict(net_state_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    # print(torch.cuda.is_available())

    model.eval()

    decode_step = len(sentence)
    predict_step = int(input("请输入要预测的词数:"))

    record = {}
    result = []

    print("--"*11, "句子的处理", "--"*11)

    # 先将整句话过一遍decode,可以理解为将整句话的每个词的hidden_state全部集中于最后一个词的hidden_state上，进而预测最后一个词的下一个词
    for i in range(decode_step):
        words = torch.LongTensor([word2id_l[i]]).to(device)
        output, record = model.decode(words, record)
        print("下一个可能的词的tensor为:{}".format(output))

    print("--"*11, "后续词预测", "--"*11)

    # 在经过整句话的decode处理后，从整句话预测的最后的一个词的下一个词开始预测
    for i in range(decode_step-1, predict_step):
        words = torch.LongTensor([word2id_l[i]]).to(device)  # 每次取一个词的索引
        output, record = model.decode(words, record)  # 解码获得最可能的下一个词
        print("下一个可能的词的tensor为:{}".format(output))
        word2id_l.append(output.item())  # 将获得的词的索引加入word2id_l继续处理

        result.append(output.item())

    print("--"*11, "预测的结果", "--"*11)
    result_words = id2word(en2index, result)
    print("预测的后续{}个词汇为:{}".format(len(result_words), result_words))
    print("预测句子结果为:{}".format(" ".join(result_words).replace("@@ ", "")))
