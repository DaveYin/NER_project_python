# encoding=utf-8

import io
import numpy as np
from gensim.models import word2vec


def create_labled_data(filename, model, tagDic, num_step, isTraining):
    f = io.open(filename, encoding="utf-8")
    data = []
    chars, tags = [], []
    ne_flag = 0
    # ne_pro = 0.05
    # 是用于训练的数据
    num = 0
    for line in f:
        if line.strip():
            items = line.strip().split()
            if len(items) != 2:
                continue
            # print items
            if items[0] in model.vocab.keys():
                chars.append(model[items[0]].tolist())
                if items[1] != u"O" and isTraining:
                    ne_flag += 1
            else:
                chars.append(model[u'<none>'].tolist())

            tags.append(tagDic.get(items[1], len(tagDic)))
            # 提取标签对应的索引号，一个标签对应唯一一个索引
            num += 1
        else:
            if len(chars) > 0:
                assert len(chars) == len(tags)
                if ne_flag < 0 and isTraining:
                    pass
                else:
                    jack = 0
                    if np.mod(num, num_step) == 0:
                        jack = num / num_step
                        for i in range(jack):
                            data.append((chars[i * num_step:(i + 1) * num_step],
                                         tags[i * num_step:(i + 1) * num_step], num_step))
                    else:
                        jack = num / num_step + 1
                        for i in range(jack - 1):
                            data.append((chars[i * num_step:(i + 1) * num_step],
                                         tags[i * num_step:(i + 1) * num_step], num_step))

                        data.append((chars[(jack - 1) * num_step:] +
                                    [[0] * len(model[u'<none>'].tolist())] * (num_step - np.mod(num, num_step)),
                                    tags[(jack - 1) * num_step:] + [len(tagDic)] * (num_step - np.mod(num, num_step)),
                                    np.mod(num, num_step)))
                        # print "22:", len(chars + [model[u'<none>'].tolist()] * (num_step - num))
            chars, tags = [], []
            ne_flag = 0
            num = 0
    f.close()
    ne_sum_length = 0
    ne_num = 0
    ne_length = 0
    ne_max_length = 0
    ne_length_dic = {}
    for cell in data:
        (_, tag_list, _) = cell
        for tag in tag_list:
            if tag != len(tagDic) and tag != tagDic.get(u"O"):
                ne_length += 1
            elif ne_length != 0:
                ne_num += 1
                ne_sum_length += ne_length

                if ne_length_dic.get(ne_length) is None:
                    ne_length_dic[ne_length] = 1
                else:
                    ne_length_dic[ne_length] += 1

                if ne_length > ne_max_length:
                    ne_max_length = ne_length
                ne_length = 0
            else:
                pass

    print "当前数据集里的命名实体长度为: ", ne_length_dic
    print "当前数据集里的命名实体平均长度为: ", ne_sum_length / ne_num
    print "当前数据集里的命名实体最大长度为: ", ne_max_length

    return data