import torch
import numpy as np
import torch.nn as nn
from LSTM.LSTM_model import M_LSTM
from BPE_handle.word22id import interchange, word2id, id2word, get_word, replace_word

PATH_MODEL = "./LSTM_train_state/lstm_train_state_73000.pth"
PATH_INDEX = "F:/PyTorch学习/BPE_handle/en_index.npy"
# PATH_INDEX = "./LSTM_file/en_index.npy"

num_layers = 5


def load(srcf):
    v = np.load(srcf, allow_pickle=True).item()  # 字典的导入
    return v


if __name__ == '__main__':
    en2index = load(PATH_INDEX)
    en = get_word(en2index)

    sentence = input("请输入测试语句:").strip().split()
    # print(sentence)

    sentence = replace_word(sentence, en)
    print(sentence)

    word2id_l = word2id(en2index, sentence)
    # print(word2id)

    model = M_LSTM(len(en2index), num_layers)
    net_state_dict = torch.load(PATH_MODEL, map_location='cpu')
    model.load_state_dict(net_state_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    # print(torch.cuda.is_available())

    model.eval()

    decode_step = len(sentence)
    predict_step = 10
    h_state = None
    cell = None
    result = []

    # 先将整句话过一遍decode
    for i in range(decode_step):
        words = torch.LongTensor([word2id_l[i]]).to(device)
        output, h_state, cell = model.decode(words, h_state, cell)

    for i in range(decode_step-1, predict_step):
        words = torch.LongTensor([word2id_l[i]]).to(device)  # 每次取一个词的索引
        output, h_state, cell = model.decode(words, h_state, cell)  # 解码获得最可能的下一个词
        word2id_l.append(output.item())

        result.append(output.item())

    result_words = id2word(en2index, result)

    print("预测句子结果为:{}".format(" ".join(result_words).replace("@@ ", "")))
