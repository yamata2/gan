import json
import time

import numpy as np
import tensorflow as tf
from IPython import embed

from model import SiameseCats
from utils import write_images

flags = tf.app.flags
flags.DEFINE_integer("epoch", 5, "Epoch to train [5]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate  for adam [0.0001]")
flags.DEFINE_integer("save_interval", 5, "The number of iterations to run for saving checkpoints [5]")
flags.DEFINE_integer("sample_interval", 1, "The number of iterations to run for sampling [1]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("n_latent", 10, "The size of z")
flags.DEFINE_integer("n_class", 10, "The number of classes.")
flags.DEFINE_integer("seed", None, "The random seed.")
flags.DEFINE_integer("load_index", None, "The load index.")
flags.DEFINE_string("running_mode", 'training', "Running mode. training or sampling.")
flags.DEFINE_boolean("allow_gpu_growth", True, "True if you want Tensorflow only to allocate the gpu memory it requires. Good for debugging, but can impact performance")
flags.DEFINE_string("from_example", None, "If you want to use example data, raise this flag.")
flags.DEFINE_boolean("wgan", False, "If you want to use Wasserstein loss, raise this flag.")
flags.DEFINE_boolean("can", False, "If you want to use a CAN, raise this flag.")
flags.DEFINE_boolean("alternative_g_loss", False, "If you want to use -log(D(G(z))) instead of log(D(1-G(z))), raise this flag.")
flags.DEFINE_string("train_csv", "./train.csv", "The data list for training data.")
flags.DEFINE_string("valid_csv", None, "The data list for validation data.")
#flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
#flags.DEFINE_float("lambda_val", 1.0, "determines the relative importance of style ambiguity loss [1.0]")
#flags.DEFINE_float("smoothing", 0.9, "Smoothing term for discriminator real (class) loss [0.9]")
#flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
##flags.DEFINE_boolean("replay", True, "True if using experience replay [True]")
#flags.DEFINE_boolean("use_resize", False, "True if resize conv for upsampling, False for fractionally strided conv [False]")
FLAGS = flags.FLAGS
for f in FLAGS:
    print f, getattr(FLAGS, f)


def batch_reader(csv_file):
    filename_queue = tf.train.string_input_producer([csv_file])
    reader = tf.TextLineReader()
    key, val = reader.read(filename_queue)
    fname, label = tf.decode_csv(val, [["aa"], [1]])
    image = tf.read_file(fname)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, (64,64),
                                   align_corners=False)
    image = image / 255.
    min_after_dequeue = FLAGS.batch_size*10
    capacity = min_after_dequeue + 3 * FLAGS.batch_size
    x, c = tf.train.shuffle_batch([image, label],
                                  batch_size=FLAGS.batch_size,
                                  capacity=capacity,
                                  min_after_dequeue=min_after_dequeue)
    return x, c

def data_from_example(example):
    if example == "mnist":
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
        x = tf.placeholder(tf.float32, [FLAGS.batch_size, 28, 28, 1])
        c = tf.placeholder(tf.int64, [FLAGS.batch_size])
        return mnist, x, c
        
def train(model):
    is_train = tf.placeholder(tf.bool, shape=(), name="phase")
    n_class = FLAGS.n_class
    
    if FLAGS.from_example == "mnist":
        n_train = 60000
        n_valid = 10000
        feeder, x, c = data_from_example("mnist")
    else:        
        n_train = 73288
        n_valid = 8158
        train_x, train_c = batch_reader(FLAGS.train_csv)
        if FLAGS.valid_csv:
            valid_x, valid_c = batch_reader(FLAGS.valid_csv)
            x, c = tf.cond(is_train, lambda: [train_x, train_c], lambda: [valid_x, valid_c])
        else:
            x, c = train_x, train_c

    drx, drgz, dcx, dcgz, gz = model(x, is_train)
    true_positive = tf.reduce_mean(tf.round(drx))*100.
    true_negative = tf.reduce_mean(1 - tf.round(drgz))*100.
    class_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(dcx, axis=1), c),
                                            tf.float32))*100.
    entropy = tf.reduce_mean(tf.reduce_sum(-dcgz*tf.log(dcgz), axis=1))
    
    with tf.name_scope('loss'):
        c_onehot = tf.one_hot(c, n_class)
        if FLAGS.wgan:
            d_loss = -tf.reduce_mean(drx) + tf.reduce_mean(drgz)
            g_loss = -tf.reduce_mean(drgz)
        else:
            d_loss = -tf.reduce_mean(tf.log(drx))\
                     -tf.reduce_mean(tf.log(1. - drgz))
            if FLAGS.alternative_g_loss:
                g_loss = tf.reduce_mean(-tf.log(drgz))
            else:
                g_loss = tf.reduce_mean(tf.log(1. - drgz))

        if FLAGS.can:
            d_loss += -tf.reduce_mean(tf.reduce_sum(c_onehot*tf.log(dcx), axis=1))
            g_loss += -tf.reduce_mean(tf.reduce_sum(1./float(n_class)*tf.log(dcgz)
                                                    + (1.-1./float(n_class))*tf.log(1.-dcgz),
                                                    axis=1))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):      
        d_var_list = model.discriminator.vars + model.discriminator_head.vars
        g_var_list = model.generator.vars

        if FLAGS.wgan:
            d_opt = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)
            g_opt = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)
        else:
            d_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            g_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)    
        d_gv = d_opt.compute_gradients(d_loss, d_var_list)
        d_step = d_opt.apply_gradients(d_gv)
        g_gv = g_opt.compute_gradients(g_loss, g_var_list)
        g_step = g_opt.apply_gradients(g_gv)
        
        if FLAGS.wgan:
            d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_var_list]


    tf.summary.image("samples", gz, max_outputs = 20)
    tf.summary.scalar("true_positive", true_positive)
    tf.summary.scalar("true_negative", true_negative)
    tf.summary.scalar("class_accuracy", class_accuracy)
    tf.summary.scalar("entropy", entropy)
    tf.summary.scalar("discriminator_loss", d_loss)
    tf.summary.scalar("generator_loss", g_loss)            
    w_list = [v for v in tf.global_variables() if "weights" in v.name]
    for w in w_list:
        tf.summary.histogram(w.name, w)
    b_list = [v for v in tf.global_variables() if "biases" in v.name]
    for b in b_list:
        tf.summary.histogram(b.name, b)
    gw_list = [gv[0] for gv in d_gv if "weights" in gv[1].name and gv[0] is not None] +\
              [gv[0] for gv in g_gv if "weights" in gv[1].name and gv[0] is not None]
    for gw in gw_list:
        tf.summary.histogram(gw.name, gw)
    gb_list = [gv[0] for gv in d_gv if "biases" in gv[1].name and gv[0] is not None] +\
              [gv[0] for gv in g_gv if "biases" in gv[1].name and gv[0] is not None]
    for gb in gb_list:
        tf.summary.histogram(gb.name, gb)
    summary = tf.summary.merge_all()
            
    ### make session
    gpuConfig = tf.ConfigProto(
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    per_process_gpu_memory_fraction = 0.5))

    sess = tf.Session(config=gpuConfig)
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('summary', sess.graph) 
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    saver = tf.train.Saver(tf.global_variables())

    itersize = n_train / FLAGS.batch_size

    previous_time = time.time()

    if FLAGS.wgan:
        FLAGS.epoch *= 6
        itersize /= 6

    def make_train_feed_dict(phase):
        if FLAGS.from_example == "mnist":
            batch_x, batch_c = feeder.train.next_batch(FLAGS.batch_size)
            batch_x = np.reshape(batch_x, [-1, 28, 28, 1])
            return {is_train: phase, x: batch_x, c:batch_c}
        else:
            return {is_train: phase}
        
    for step in range(FLAGS.epoch):
        error = 0.
        for itr in range(itersize):
            if FLAGS.wgan:
                for _ in range(5):
                    feed_dict = make_train_feed_dict(True)
                    _d, dl = sess.run([d_step, d_loss], feed_dict=feed_dict)
                    sess.run(d_clip, feed_dict={is_train: True})
                feed_dict = make_train_feed_dict(True)
                _g, gl = sess.run([g_step, g_loss], feed_dict=feed_dict)
            else:
                feed_dict = make_train_feed_dict(True)
                _d, _g, dl, gl = sess.run([d_step, g_step, d_loss, g_loss], feed_dict=feed_dict)
                    
            current_time = time.time()
            #print "epoch:{}...{}/{} d_loss:{}, g_loss:{}, time:{}".format(step, itr, itersize, dl, gl, current_time-previous_time)
            previous_time = current_time

        if FLAGS.from_example == "mnist":
            batch_x, batch_c = feeder.test.next_batch(FLAGS.batch_size)
            batch_x = np.reshape(batch_x, [-1, 28, 28, 1])
            feed_dict = {is_train: False, x: batch_x, c:batch_c}
        else:
            feed_dict = {is_train: False}

        tp, tn, ca, dl, gl, e, samples, s = sess.run([true_positive, true_negative, class_accuracy,
                                                      d_loss, g_loss, entropy, gz, summary],
                                                     feed_dict=feed_dict)
        writer.add_summary(s, step)
        current_time = time.time()
        in_pix_std = np.mean(np.std(samples, axis=0))
        print "epoch:{}...validation d_loss:{} g_loss:{} true positive: {}%, true negative: {}%, class accuracy: {}%, entropy:{}, std_per_pixel: {} time:{}".format(step, dl, gl, tp, tn, ca, e, in_pix_std, current_time-previous_time)
        previous_time = current_time        
        

        
        if (step + 1) % FLAGS.save_interval == 0:
            saver.save(sess, "{}_{}".format("checkpoints/model", step))
        if (step + 1) % FLAGS.sample_interval == 0:
            write_images(samples*255., "./samples/epoch_%.4d/" %step)
            
def sample(model):
    samples, drgz, dcgz = model.sample(False)

    ### make session
    gpuConfig = tf.ConfigProto(
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    per_process_gpu_memory_fraction = 0.5))
    sess = tf.Session(config=gpuConfig)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, "./checkpoints/model_{}".format(FLAGS.load_index))
    DRGZ, DCGZ, s = sess.run([drgz, dcgz, samples])
    write_images(s*255., "./samples/manual")
    np.savetxt("./samples/drgz.txt", DRGZ)
    np.savetxt("./samples/dcgz.txt", DCGZ)


def main():
    tf.reset_default_graph()
    if FLAGS.seed is not None:
        np.random.seed(FLAGS.seed)
        tf.set_random_seed(FLAGS.seed)
    
    if FLAGS.from_example == "mnist":
        g_layers = json.load(open("./mnist_gen_conf.json", "r"))
        d_layers = json.load(open("./mnist_dis_conf.json", "r"))
    else:
        g_layers = json.load(open("./generator_conf.json", "r"))
        d_layers = json.load(open("./discriminator_conf.json", "r"))

    model = SiameseCats(g_layers,
                        d_layers,
                        FLAGS.batch_size,
                        FLAGS.n_latent,
                        FLAGS.n_class,
                        FLAGS.wgan)

    if FLAGS.running_mode == "training":
        train(model)
        if FLAGS.running_mode == "sampling":
            sample(model)
            
if __name__ == "__main__":
    main()
