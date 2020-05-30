import numpy as np
import linecache
from Dataset.dataset import Dataset
from Model.model import Model
from Trainer.trainer import Trainer
from Trainer.pretrainer import PreTrainer
from Utils import gpu_info
from Utils.utils import mkdir
import os
import random
import tensorflow as tf
from Model.mModel import MVModel
import scipy.io as sio

if __name__ == '__main__':

    # gpus_to_use, free_memory = gpu_info.get_free_gpu()
    # print( gpus_to_use, free_memory)
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use
    dataset_name = 'cora'
    layers = [200,100]
    View_num = 2
    beta_W = 10
    learning_rate=1e-5
    random.seed(9001)
    dataset_config = {
        'View': ['./Database/' + dataset_name + '/View1.txt', './Database/' + dataset_name + '/edges.txt'],
        'label_file': './Database/' + dataset_name + '/group.txt'}

    graph = Dataset(dataset_config)

    dims = [np.shape(vData)[1] for vData in graph.ViewData]

    pretrain_config = {
        'View': layers,
        'pretrain_params_path': './Log/' + dataset_name + '/pretrain_params.pkl'}

    if False:
        pretrainer = PreTrainer(pretrain_config)
        for i in range(len(graph.ViewData)):
            pretrainer.pretrain(graph.ViewData[i], 'V' + str(i + 1))

    model_config = {
        'weight_decay':10.0,
        'View_num': View_num,
        'View': layers,
        'is_init': True,
        'pretrain_params_path': './Log/' + dataset_name + '/pretrain_params.pkl'
    }

    pathResult = './result/' + dataset_name + "/"
    mkdir(pathResult)
    with open(pathResult + dataset_name + "_" + str(learning_rate) + '_' + str(beta_W) + '.txt', "w") as f:
        # for beta_i in np.transpose([1,10,50,100,200]):
        #     for alpha_i in [0.001,0.01,0.1,1,10,100]:
        #         for gama_i in [0.001,0.01,0.1,1,10,100]:
        for ccsistent_loss in [0.001, 0.01, 0.1, 0, 1, 10, 50, 100, 200]:
            tf.reset_default_graph()
            trainer_config = {
                'ccsistent_loss': ccsistent_loss,
                'beta_W': beta_W,
                'View_num': View_num,
                'View': layers,
                'dims': dims,
                'drop_prob': 0.2,
                'learning_rate':learning_rate,
                'batch_size': 128,
                'num_epochs': 1000,
                'model_path': './Log/' + dataset_name + '/' + dataset_name + '_model.pkl',
            }

            model = MVModel(model_config)
            trainer = Trainer(model, trainer_config)
            trainer.train(graph)
            acc, nmi = trainer.inferCluster(graph)
            result_single = 'ccsistent_loss={:.4f}:'.format(ccsistent_loss) + ' acc={:.4f}'.format(
                acc) + ' & ' + 'nmi={:.4f}'.format(nmi)
            f.write(result_single + '\n')
            f.flush()

