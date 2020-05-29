import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from Utils.utils import *
from Model.SingleAE import SingleAE



class Trainer(object):
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.dims = config['dims']
        self.View_num=config['View_num']

        self.View = config['View']
        self.drop_prob = config['drop_prob']

        self.beta_W = config['beta_W']
        self.ccsistent_loss=config['ccsistent_loss']

        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.model_path = config['model_path']

        self.vList=[]
        for i in range(self.View_num):
            self.vList.append(tf.placeholder(tf.float32, [None, self.dims[i]],name='V'+str(i+1)))

        self.mvList = self.model.getModel()

        # 如果输入进来是相似性矩阵，我们则不需要进行距离计算，如果输入进来时原始数据，我们则需要进行计算
        self.optimizer, self.loss = self._build_training_graph()

        _, self.H = self._build_eval_graph()

        gpu_config = tf.ConfigProto()

        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def calc_sig(self, DD):
        one = tf.ones_like(DD)
        zero = tf.zeros_like(DD)

        # 如果大于kth则为1，否则为0
        TFMat = tf.where(DD <= 0, x=zero, y=one)

        return TFMat;

    def get_2nd_loss(self, X, newX, beta=5):
        TFMat = self.calc_sig(X)
        TFMat = X
        B = TFMat * (beta - 1) + 1

        return tf.reduce_sum(tf.pow((newX - X) * B, 2), 1)

    def _build_training_graph(self):

        netList=[]
        reconList = []

        for i in range(self.View_num):
            net_V1, V1_recon=self.mvList[i].forward(self.vList[i], drop_prob=self.drop_prob,view=str(i+1), reuse=False)
            netList.append(net_V1)
            reconList.append(V1_recon)
        loss=0
        if self.beta_W == 0:
            for i in range(self.View_num):
                loss+= tf.reduce_mean(tf.reduce_sum(tf.square(self.vList[i] - reconList[i]), 1))
        else:
            for i in range(self.View_num):
                loss+=  tf.reduce_mean(self.get_2nd_loss(self.vList[i], reconList[i], self.beta_W))


        consiste_loss=tf.reduce_mean(tf.reduce_sum(tf.square(netList[0] - netList[1]), 1))

        loss = loss+self.ccsistent_loss*consiste_loss


        vars_net = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V1_encoder')
        vars_att = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V2_encoder')




        print(vars_net)

        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=vars_net + vars_att)

        return opt, loss

    def _build_eval_graph(self,fuse_type="cat"):
        netList=[]
        for i in range(self.View_num):
            net_V, _ = self.mvList[i].forward(self.vList[i], drop_prob=0.0,view=str(i+1), reuse=True)
            netList.append(net_V)

        if(fuse_type=="cat"):
            H = tf.concat([tf.nn.l2_normalize(net, dim=1) for net in netList], axis=1)
        else:
            pass
        return netList, H

    def train(self, graph):
        
        for epoch in range(self.num_epochs):
            idx1 = self.generate_samples(graph)
            index = 0
            cost = 0.0
            cnt = 0
            while True:
                if index > graph.num_nodes:
                    break
                if index + self.batch_size < graph.num_nodes:
                    mini_batch1 = graph.sample_by_idx(idx1[index:index + self.batch_size])
                else:
                    mini_batch1 = graph.sample_by_idx(idx1[index:])

                feed_dict = {}
                for i in range(self.View_num):
                    feed_dict["V" + str(i + 1) + ":0"] = mini_batch1["V" + str(i + 1)]
                index += self.batch_size
                loss, _ = self.sess.run([self.loss, self.optimizer],feed_dict=feed_dict)
                cost += loss
                cnt += 1

                if graph.is_epoch_end:
                    break
            cost /= cnt

            if epoch % 50 == 0:
                train_emb = None
                train_label = None
                while True:
                    mini_batch = graph.sample(self.batch_size, do_shuffle=False, with_label=True)
                    feed_dict={}
                    for i in range(self.View_num):
                        feed_dict["V"+str(i+1)+":0"]=mini_batch["V"+str(i+1)]
                    emb = self.sess.run(self.H,feed_dict=feed_dict)
                    if train_emb is None:
                        train_emb = emb
                        train_label = mini_batch.Y
                    else:
                        train_emb = np.vstack((train_emb, emb))
                        train_label = np.vstack((train_label, mini_batch.Y))

                    if graph.is_epoch_end:
                        break

                acc, nmi = node_clustering(train_emb, train_label)

                print('Epoch-{},loss: {:.4f}, acc {:.4f}, nmi {:.4f}'.format(epoch, cost, acc, nmi))

        self.save_model()

    
    def inferCluster(self, graph):
        self.sess.run(tf.global_variables_initializer())
        self.restore_model()
        print("Model restored from file: %s" % self.model_path)

        train_emb = None
        train_label = None
        while True:
            mini_batch = graph.sample(self.batch_size, do_shuffle=False, with_label=True)

            feed_dict = {}
            for i in range(self.View_num):
                feed_dict["V" + str(i + 1) + ":0"] = mini_batch["V" + str(i + 1)]

            emb = self.sess.run(self.H, feed_dict=feed_dict)

            if train_emb is None:
                train_emb = emb
                train_label = mini_batch.Y
            else:
                train_emb = np.vstack((train_emb, emb))
                train_label = np.vstack((train_label, mini_batch.Y))

            if graph.is_epoch_end:
                break

        acc, nmi = node_clustering(train_emb, train_label)

        print(' acc {:.4f}, nmi {:.4f}'.format(acc, nmi))
        return acc, nmi

    def generate_samples(self, graph):

        order = np.arange(graph.num_nodes)
        np.random.shuffle(order)

        return order

    def save_model(self):
        self.saver.save(self.sess, self.model_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
