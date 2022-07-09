import torch
import h5py
import torch.nn as nn
import torch.optim as op
from LSTM.LSTM_model import M_LSTM

# hdf5文件路径
PATH_F5 = "F:/PyTorch学习/BPE_handle/result.hdf5"
# PATH_F5 = "./RNN_file/result.hdf5"

# hdf5文件读出及词典大小获取
f = h5py.File(PATH_F5, "r")
vocab_size = f["nword"][()]  # 取出主键为nword的所有键值，即收集的词典大小（读取的数据是标量，用[()]）
# print(vocab_size)

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 相关参数设置
num_epoch = 3  # 训练轮次
curb = 0  # 起始组数


if __name__ == '__main__':
    # 固定随机种子
    torch.manual_seed(0)  # 设置CPU生成随机数的种子，为确保每次生成固定的随机数，方便下次复现实验结果，需在模型定义前设置

    # 模型定义，并传入词典大小vocab_size
    model = M_LSTM(vocab_size)
    model = model.to(device)
    model.train()

    # 参数初始化
    # 服从均匀分布U(−a,a)，par应该是2维及以上
    # 用以保证初始化的值不会因其大小而在层数的传递时导致方差变化，使通过每一层网络时保证输入和输出的方差相同
    for par in model.parameters():  # 初始化的参数包括权重weight，偏置值bias
        if par.dim() > 1:
            nn.init.xavier_uniform_(par, gain=1)

    # 创建损失函数，使用交叉熵代价函数
    # nn.CrossEntropyLoss()结合了nn.log_softmax()和nn.NLLLoss()
    # reduction='sum'表示将loss值求和
    loss_f = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    loss_f.to(device)

    # 创建优化器
    opt = op.Adam(model.parameters())

    # 训练过程
    for epoch in range(num_epoch):
        print('**' * 15, "训练轮次{}".format(epoch + 1), '**' * 15)
        curb = 0

        for index in f["group"]:
            data = torch.LongTensor(f["group"][index][:])
            # print(data)
            data_num = data.shape[1]
            # print(data_num)
            input = data.narrow(1, 0, data_num - 1)
            # print(input)
            # print(len(input))
            target = data.narrow(1, 1, data_num - 1)
            # print(target)
            # print(len(target))

            input = input.to(device)
            target = target.to(device)

            # forward
            output = model(input)
            output = output.to(device)
            loss = loss_f(output.transpose(1, 2), target)  # 输入的tensor需要为(minibatch,C)格式，因此需要将output中的1、2维转置
            # loss.requires_grad_(True)

            # backward
            loss.backward()  # 梯度

            # 更新及初始化参数梯度
            if curb % 10 == 0:
                opt.step()  # 更新参数
                opt.zero_grad()  # 参数梯度初始化为0

            # 每100个batch打印一次loss值
            if curb % 100 == 0:
                print('loss: {:.6f}  [batch_num:{}]'.format(loss / vocab_size, curb))
                print('--' * 30)

            # 每处理1000个batch保存一次模型参数
            if curb % 1000 == 0:
                torch.save(model.state_dict(), './LSTM_train_state/train_state_{}.pth'.format(curb))
                print('模型参数保存成功')
                print('--' * 30)
            curb += 1
    f.close()