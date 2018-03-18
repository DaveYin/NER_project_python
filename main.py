# coding=utf-8

import argparse
import sys
import myio
import cPickle
import gensim
from util import *


def main(args):
    # 加载字向量模型
    model_embedding = gensim.models.KeyedVectors.load_word2vec_format(args.embeddings, binary=False)

    if args.action == "evaluate":
        # 获取tagDic和args等数据
        print "加载模型..."
        loaded_model = cPickle.load(open(args.model))
        # 加载参数
        loaded_args = loaded_model["args"]
        print "模型参数: " + str(loaded_args)
        tagDic = loaded_model["tagDic"]
        print "标签字典: " + str(tagDic)
        transition_params = loaded_model["trans_params"]
        print "状态转移矩阵: " + str(transition_params)
        # ===============================================
        # 得到测试集数据
        test_data = myio.create_labled_data(args.test, model_embedding, tagDic, loaded_args.train_num_step, False)
        print "得到了测试集数据!"
        randomdata(test_data)
        sentence_embedding = []  # 句子
        sentence_label = []
        sentence_length = []
        test_queue = []  # 总数据(x, y)
        for i in range(len(test_data) // loaded_args.eval_batch_size):
            for j in range(loaded_args.eval_batch_size):
                (x, y, z) = test_data[loaded_args.eval_batch_size * i + j]
                sentence_embedding.append(x)
                sentence_label.append(y)
                sentence_length.append(z)
            test_queue.append((sentence_embedding, sentence_label, sentence_length))
            sentence_embedding = []  # 清空
            sentence_label = []  # 清空
            sentence_length = []
        print "得到了预处理后的测试集数据:)"
        with tf.Session() as session:
            new_saver = tf.train.import_meta_graph(args.score_dir + '/my-model-30.meta')
            new_saver.restore(session, args.score_dir + '/my-model-30')
            # 得到输出
            eval_model_loss = tf.get_collection('eval_model_loss')[0]
            eval_model_tags_scores = tf.get_collection('eval_model_tags_scores')[0]
            eval_model_transition_params = tf.get_collection('eval_model_transition_params')[0]
            eval_model_sequence_lengths = tf.get_collection('eval_model_sequence_lengths')[0]

            graph = tf.get_default_graph()
            # 得到placeholder
            eval_model_input_data = graph.get_operation_by_name('language_model/input_data_1').outputs[0]
            eval_model_input_data_lengths = graph.get_operation_by_name('language_model/input_data_lengths_1').outputs[0]
            eval_model_targets = graph.get_operation_by_name('language_model/targets_1').outputs[0]
            eval_model_trans_params = graph.get_operation_by_name('language_model/trans_params_1').outputs[0]

            print "测试数据有", len(test_queue), "组"
            # 测试集数据进行评估
            total_labels = 0
            total_reco_ens = 0
            correct_nes = 0
            total_nes = 0
            final_ne_result = []
            # Start input enqueue threads.
            coord = tf.train.Coordinator()  # Create a new Coordinator for threads.
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            # Starts all queue runners collected in the graph.
            # It returns the list of all threads.
            for index in range(len(test_queue)):
                (x, y, z) = test_queue[index]  # data -- 每一个元素代表一次训练
                # print "step ", index + 1, ": 在当前batch上运行train_op"
                loss, tf_tags_scores, _, sequence_lengths, _, = session.run(
                    # 对应的输出，知道tf_tags_scores就相当于知道了最终结果。tf_transition_params是转移矩阵，代表无向图。
                    [eval_model_loss, eval_model_tags_scores, eval_model_transition_params, eval_model_sequence_lengths,
                     tf.no_op()],
                    {eval_model_input_data: np.array(x), eval_model_targets: np.array(y),
                     eval_model_trans_params: transition_params, eval_model_input_data_lengths: np.array(z)}  # 需要的输入
                )

                for tf_tags_scores_, y_, sequence_length_ in zip(tf_tags_scores, np.array(y), sequence_lengths):
                    # Remove padding from the scores and tag sequence.
                    tf_tags_scores_ = tf_tags_scores_[:sequence_length_]
                    y_ = y_[:sequence_length_]
                    # print tf_output
                    # Compute the highest scoring sequence.
                    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_tags_scores_, transition_params)
                    final_ne_result.append(viterbi_sequence)  # 存储输出的NE结果
                    viterbi_sequence_array = np.array(viterbi_sequence)
                    # print viterbi_sequence_array
                    # print y_
                    # Evaluate word-level precision, recall, F-measure.
                    # 总的标签数
                    total_labels += sequence_length_
                    # 总的识别实体数
                    total_reco_ens += sequence_length_ - np.sum(
                        np.equal(viterbi_sequence_array, np.ones(viterbi_sequence_array.shape) * len(tagDic))
                    ) - np.sum(
                        np.equal(viterbi_sequence_array, np.ones(viterbi_sequence_array.shape) * tagDic.get(u"O"))
                    )
                    # 正确识别的实体数
                    correct_nes += np.sum(
                        np.logical_and(
                            np.logical_and(
                                np.equal(viterbi_sequence_array, y_),
                                np.logical_not(np.equal(viterbi_sequence_array,
                                                        np.ones(viterbi_sequence_array.shape) * len(tagDic)))
                            ),
                            np.logical_not(np.equal(viterbi_sequence_array,
                                                    np.ones(viterbi_sequence_array.shape) * tagDic.get(u"O")))
                        )
                    )
                    # 总的实体数
                    total_nes += sequence_length_ - np.sum(
                        np.equal(y_, np.ones(y_.shape) * len(tagDic))
                    ) - np.sum(
                        np.equal(y_, np.ones(y_.shape) * tagDic.get(u"O"))
                    )
            # When done, ask the threads to stop.
            coord.request_stop()  # Request that the threads stop.
            coord.join(threads)  # Wait for threads to terminate.

            precision = 100.0 * correct_nes / float(total_reco_ens + 1e-8)
            recall = 100.0 * correct_nes / float(total_nes + 1e-8)
            F_measure = 2 * precision * recall / (precision + recall + 1e-8)
            # print("precision: %.2f%%;  recall: %.2f%%;  F_measure: %.2f%%" % (precision, recall, F_measure))
            # 一轮完成后输出一个准确率、召回率、F值（这个值是累积的）
            print("测试集——precision: %.2f%%;  recall: %.2f%%;  F_measure: %.2f%%" % (precision, recall, F_measure))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    # 输入的系统需要的参数
    argparser.add_argument("action",
                           type=str,
                           default="train"
                           )

    # 训练后的最佳模型
    argparser.add_argument("--model",
                           type=str,
                           default=""
                           )

    # data for training
    argparser.add_argument("--train",
                           type=str,
                           default="./workDir/data/traindata"
                           )

    # data for test
    argparser.add_argument("--test",
                           type=str,
                           default="./workDir/data/testdata"
                           )

    # data for development
    argparser.add_argument("--dev",
                           type=str,
                           default="./workDir/data/devdata"
                           )

    # data for evaluation / here, testdata is used to evaluated.
    argparser.add_argument("--eva",
                           type=str,
                           default="./workDir/data/testdata"
                           )

    # data for extracting entities
    argparser.add_argument("--corpus",
                           type=str,
                           default="./workDir/data/rawdata"
                           )

    # output result of NER
    argparser.add_argument("--output",
                           type=str,
                           default="./result"
                           )

    # character embedding
    argparser.add_argument("--embeddings",
                           type=str,
                           default="./char_embedding.txt"
                           )

    # size of neural network
    argparser.add_argument("--hidden_size",
                           type=int,
                           default=150
                           )

    # dropout 的概率p
    argparser.add_argument("--drop_pro",
                           type=float,
                           default=0.5
                           )

    # 学习速率
    argparser.add_argument("--lrate",
                           type=float,
                           default=0.02
                           )

    # attention window
    argparser.add_argument("--attn_length",
                           type=int,
                           default=10
                           )

    # 规则化参数
    argparser.add_argument("--re_scale",
                           type=float,
                           default=0.1)

    # momentum
    argparser.add_argument("--momentum",
                           type=float,
                           default=0.9)

    # 训练数据batch的大小
    argparser.add_argument("--train_batch_size",
                           type=int,
                           default=20
                           )

    # 测试数据batch的大小
    argparser.add_argument("--eval_batch_size",
                           type=int,
                           default=1
                           )

    # 训练数据截断长度
    argparser.add_argument("--train_num_step",
                           type=int,
                           default=140
                           )

    # 测试数据截断长度
    argparser.add_argument("--eval_num_step",
                           type=int,
                           default=1
                           )

    # 评分文件夹
    argparser.add_argument("--score_dir",
                           type=str,
                           default="./workDir/trainResult"
                           )

    # 训练轮数
    argparser.add_argument("--epoch",
                           type=int,
                           default=50
                           )

    # LSTM层数
    argparser.add_argument("--num_layers",
                           type=int,
                           default=2
                           )

    args = argparser.parse_args()
    main(args)