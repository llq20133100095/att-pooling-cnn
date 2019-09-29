from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import logging
import os
import time

import configConll04
import utils
from model import Model
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix, precision_score
from sklearn.preprocessing import OneHotEncoder


config = configConll04.FLAGS

def run_epoch(session, model, batch_iter, is_training=True, verbose=True):
  start_time = time.time()
  acc_count = 0
  step = 0#len(all_data)
  predict_count=np.array([0])
  groundtrue_y_count=np.array([0])
  all_distance_count = np.reshape(np.array([0] * config.classnum), (1, -1))

  for batch in batch_iter:
    step += 1
    batch = (x for x in zip(*batch))
    sents, relations, e1, e2, dist1, dist2 = batch
    # sents is a list of np.ndarray, convert it to a single np.ndarray
    sents = np.vstack(sents)

    in_x, in_e1, in_e2, in_dist1, in_dist2, in_y = model.inputs
    feed_dict = {in_x: sents, in_e1: e1, in_e2: e2, in_dist1: dist1, 
                 in_dist2: dist2, in_y: relations}
    if is_training:
      _, _, acc, loss = session.run([model.train_op, model.reg_op, model.acc, model.loss], feed_dict=feed_dict)
      acc_count += acc
      if verbose and step%10 == 0:
        logging.info("  step: %d acc: %.2f%% loss: %.2f time: %.2f" %(
          step,
          acc_count / (step * config.batch_size) * 100,
          loss,
          time.time() - start_time
          ))
    else:
      acc, predict, groundtrue_y, all_distance = session.run([model.acc, model.predict, model.in_y, model.all_distance], feed_dict=feed_dict)
      acc_count += acc
      predict_count=np.concatenate((predict_count,predict))
      groundtrue_y_count=np.concatenate((groundtrue_y_count,groundtrue_y))
      all_distance_count = np.concatenate((all_distance_count, all_distance), axis = 0)

  return acc_count / (step * config.batch_size), predict_count, groundtrue_y_count, all_distance_count

  

def init():
  path = config.data_path
  config.embedding_file = os.path.join(path, config.embedding_file)
  config.embedding_vocab = os.path.join(path, config.embedding_vocab)
  config.train_file = os.path.join(path, config.train_file)
  config.test_file = os.path.join(path, config.test_file)

  # Config log
  if config.log_file is None:
    logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
  else:
    if not os.path.exists(config.save_path):
      os.makedirs(config.save_path)
    logging.basicConfig(filename=config.log_file,
                      filemode='a', level=logging.DEBUG,
                      format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
  # Load data
  # data = (sentences, relations, e1_pos, e2_pos)
  train_data = utils.load_data(config.train_file)
  test_data = utils.load_data(config.test_file)

  logging.info('trian data: %d' % len(train_data[0]))
  logging.info('test data: %d' % len(test_data[0]))

  # Build vocab.   word_dict:Count the number of words ("word":number)
  word_dict = utils.build_dict(train_data[0] + test_data[0])
  logging.info('total words: %d' % len(word_dict))

  embeddings = utils.load_embedding(config, word_dict)

  # Log parameters
  flags = config.__dict__['__flags']
  flag_str = "\n"
  for k in flags:
    flag_str += "\t%s:\t%s\n" % (k, flags[k])
  logging.info(flag_str)

  # vectorize data
  # vec = (sents_vec, relations, e1_vec, e2_vec, dist1, dist2)
  max_len_train = len(max(train_data[0], key=lambda x:len(x)))
  max_len_test = len(max(test_data[0], key=lambda x:len(x)))
  max_len = max(max_len_train, max_len_test)
  config.max_len = max_len

  #word vector
  train_vec = utils.vectorize(train_data, word_dict, max_len)
  test_vec = utils.vectorize(test_data, word_dict, max_len)

  return embeddings, train_vec, test_vec

  
def main(_):
  embeddings, train_vec, test_vec = init()
  bz = config.batch_size

  with tf.Graph().as_default():
    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=None):
        m_train = Model(config, embeddings, is_training=True)
      # tf.summary.scalar("Training_Loss", m_train.loss)
      # tf.summary.scalar("Training_acc", m_train.acc)

    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True):
        m_test = Model(config, embeddings, is_training=False)
      # tf.summary.scalar("test_acc", m_test.acc)
    
    sv = tf.train.Supervisor(logdir=config.save_path,
                             global_step=m_train.global_step)
    with sv.managed_session() as session:
      if config.test_only:
        test_iter = utils.batch_iter(list(zip(*test_vec)), bz, shuffle=False)
        test_acc = run_epoch(session, m_test, test_iter, is_training=False)
        print("test acc: %.3f" % test_acc)
      else:
        for epoch in range(config.num_epoches):
          # lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
          # m.assign_lr(session, config.learning_rate * lr_decay)
          train_iter = utils.batch_iter(list(zip(*train_vec)), bz, shuffle=True)
          test_iter = utils.batch_iter(list(zip(*test_vec)), bz, shuffle=False)
          train_acc = run_epoch(session, m_train, train_iter, verbose=False)
          test_acc = run_epoch(session, m_test, test_iter, is_training=False)
          logging.info("Epoch: %d Train: %.2f%% Test: %.2f%%" % 
                                      (epoch + 1, train_acc*100, test_acc*100))
        if config.save_path:
          sv.saver.save(session, config.save_path, global_step=sv.global_step)

def precision_recall_f1(con_mat):
    """
    Compute the top1-precision
    """
    precision=[]
    recall=[]
    each_precision=0.0
    each_recall=0.0
    for i in range(3):
        try:
            each_precision=float(con_mat[i][i])/float(np.sum(con_mat[:,i]))   
        except:
            each_precision=0.0
        try:
            each_recall=float(con_mat[i][i])/float(np.sum(con_mat[i,:]))
        except:    
            each_recall=0.0
            
        precision.append(each_precision)
        recall.append(each_recall)
        
    precision_mean=np.mean(precision)
    recall_mean=np.mean(recall)
    f1=(2*precision_mean*recall_mean)/(precision_mean+recall_mean)
    return f1


if __name__ == '__main__':
#  tf.app.run()

  embeddings, train_vec, test_vec = init()
  bz = config.batch_size

  with tf.Graph().as_default():
    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=None):
        m_train = Model(config, embeddings, is_training=True)
      # tf.summary.scalar("Training_Loss", m_train.loss)
      # tf.summary.scalar("Training_acc", m_train.acc)

    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True):
        m_test = Model(config, embeddings, is_training=False)
      # tf.summary.scalar("test_acc", m_test.acc)
    
    sv = tf.train.Supervisor(logdir=config.save_path,
                             global_step=m_train.global_step)
    with sv.managed_session() as session:
      if config.test_only:
        test_iter = utils.batch_iter(list(zip(*test_vec)), bz, shuffle=False)
        test_acc = run_epoch(session, m_test, test_iter, is_training=False)
        print("test acc: %.3f" % test_acc)
      else:
        max_f1_batch = 0
        max_f1 = 0
        for epoch in range(config.num_epoches):
          # lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
          # m.assign_lr(session, config.learning_rate * lr_decay)
          train_iter = utils.batch_iter(list(zip(*train_vec)), bz, shuffle=True)
          test_iter = utils.batch_iter(list(zip(*test_vec)), bz, shuffle=False)
          train_acc, _, _, _ = run_epoch(session, m_train, train_iter, verbose=False)
          test_acc, predict, groundtrue_y, all_distance = run_epoch(session, m_test, test_iter, is_training=False)
          
          predict=predict[1:-14]
          groundtrue_y=groundtrue_y[1:-14]
          all_distance = all_distance[1:-14]
          
          test_con_mat=confusion_matrix(groundtrue_y,predict)
          #computer F1_Score
          test_f1_score=precision_recall_f1(test_con_mat)

          logging.info("Epoch: %d Train: %.2f%% Test: %.2f%%" % 
                                      (epoch + 1, train_acc*100, test_f1_score*100))

          if max_f1 < test_f1_score:
            max_f1 = test_f1_score
            max_f1_batch = epoch + 1

        if config.save_path:
          sv.saver.save(session, config.save_path, global_step=sv.global_step)

        #save the prediction
        one_enc = OneHotEncoder(sparse = False)
        groundtrue_y = one_enc.fit_transform(np.reshape(groundtrue_y, (-1, 1)))
        save_data = dict(
                testing_label_1hot = groundtrue_y,
                test_prediction_all = all_distance)
        np.savez('./Conll04_data/experiment/Att_pooling_CNN_Conll04.npz', **save_data)

        logging.info("Epoch: %d Max Test: %.2f%%" % (max_f1_batch, max_f1 * 100))

