# -*- coding: utf-8 -*-
import tensorflow as tf


#初始化变量和模型参数，定义训练闭环中的运算


def inference(X):
    #计算推断模型在数据X上的输出，并将结果返回
    return

def loss(X,Y):
    #依据训练数据X及期期望输出Y计算损失
    return

def inputs():
    #读取或生成训练数据X及期望输出Y
    return

def  train(total_loss):
    #依据计算的总损失训练或调整模型参数
    return

def  evaluate(sess,X,Y):
    #对训练得到的模型 进行评估
    return

#创建检查点Saver对象
saver = tf.train.Saver()
save_path = "my_model"

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    X,Y = inputs()
    
    initial_step = 0
    #验证之前是否已经保存了检查点文件
    ckpt = tf.train.get_checkpoint_state(save_path)
    if ckpt and ckpt.model_checkpoint_path:
        #从检查点恢复
        saver.restore(sess,ckpt.model_checkpoint_path)
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-',1)[1])

    total_loss = loss(X,Y)
    train_op = train(total_loss)
    coord =  tf.train.Coodinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)

    #实际训练迭代次数
    training_steps = 1000
    for step in range(initial_step,training_steps):
        sess.run(train_op)
    
    if  step % 10 == 0:
        print "loss:",sess.run([total_loss])
        saver.save(sess,save_path,global_step = step)
        

    evaluate(sess,X,Y)

    coord.request_stop()
    coord.join(threads)
    saver.save(sess,save_path,global_step = training_steps)
    sess.close()




    
