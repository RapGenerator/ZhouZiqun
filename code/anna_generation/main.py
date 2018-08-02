import time
import numpy as np
import tensorflow as tf

ANNA_DATA_FILE = 'anna.txt'
# Params
batch_size = 100
n_steps = 100
lstm_size = 512
n_layers = 2
learning_rate = 0.001
keep_prob = 0.5
epochs = 200
save_every_n = 200


def load_data():
    with open(ANNA_DATA_FILE, 'r') as f:
        return f.read()


def process_data(text):
    vocab = set(text)
    vocab_to_idx = {c: i for i, c in enumerate(vocab)}
    idx_to_vocab = {i: c for i, c in enumerate(vocab)}
    return vocab, vocab_to_idx, idx_to_vocab


def get_batches(arr, n_seqs, n_steps):
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)

    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n + n_steps]
        y = np.zeros_like(x)
        y[:, :-1] = x[:, 1:]
        y[:, -1] = x[:, 0]
        yield x, y


def build_inputs(n_seqs, n_steps):
    inputs = tf.placeholder(tf.int32, shape=(n_seqs, n_steps), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(n_seqs, n_steps), name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, keep_prob


def build_lstm(lstm_size, n_layers, batch_size, keep_prob):
    def lstm_cell():
        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
        return tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)

    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(n_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state


def build_output(lstm_output, in_size, out_size):
    seq_output = tf.concat(lstm_output, 1)
    x = tf.reshape(seq_output, [-1, in_size])
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    logits = tf.matmul(x, softmax_w) + softmax_b
    out = tf.nn.softmax(logits, name='predictions')

    return out, logits


def build_loss(logits, targets, lstm_size, num_classes):
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)

    return loss


def build_optimizer(loss, learning_rate, grad_clip):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer


def pick_top_n(preds, vocab_size, top_n=5):
    """
    从预测结果中选取前top_n个最可能的字符

    preds: 预测结果
    vocab_size
    top_n
    """
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample(vocab, vocab_to_int, int_to_vocab, checkpoint, n_samples,
           lstm_size, vocab_size, prime="The "):
    # 将输入的单词转换为单个字符组成的list
    samples = [c for c in prime]
    # sampling=True意味着batch的size=1 x 1
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        x = np.zeros((1, 1))
        # 加载模型参数，恢复训练
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        # 添加字符到samples中
        samples.append(int_to_vocab[c])

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x[0, 0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])

    return ''.join(samples)


def train(vocab, encoded):
    # Create model
    model = CharRNN(n_classes=len(vocab), batch_size=batch_size, n_steps=n_steps,
                    lstm_size=lstm_size, n_layers=n_layers,
                    learning_rate=learning_rate)
    saver = tf.train.Saver(max_to_keep=100)

    # Train model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        counter = 0
        for e in range(epochs):
            # Train network
            new_state = sess.run(model.initial_state)
            for x, y in get_batches(encoded, batch_size, n_steps):
                counter += 1
                start = time.time()
                feed = {
                    model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state
                }
                batch_loss, new_state, _ = sess.run([
                    model.loss,
                    model.final_state,
                    model.optimizer
                ], feed_dict=feed)
                end = time.time()

                # Print
                if counter % 500 == 0:
                    print('epoch: {}/{}'.format(e + 1, epochs),
                          'step:  {}'.format(counter),
                          'loss:  {}'.format(batch_loss),
                          'speed: {:.4f} sec/batch'.format(end - start))

                if counter % save_every_n == 0:
                    saver.save(sess, 'checkpoint/i{}_l{}.ckpt'.format(counter, lstm_size))
        # Save final result
        saver.save(sess, 'checkpoint/i{}_l{}.ckpt'.format(counter, lstm_size))


class CharRNN:
    def __init__(self, n_classes, batch_size=64, n_steps=50,
                 lstm_size=128, n_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False):
        if sampling == True:
            batch_size, n_steps = 1, 1
        else:
            batch_size, n_steps = batch_size, n_steps

        tf.reset_default_graph()

        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, n_steps)
        cell, self.initial_state = build_lstm(lstm_size, n_layers, batch_size, self.keep_prob)
        x_one_hot = tf.one_hot(self.inputs, n_classes)

        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state

        self.prediction, self.logits = build_output(outputs, lstm_size, n_classes)

        self.loss = build_loss(self.logits, self.targets, lstm_size, n_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)


def main():
    # Load data
    text = load_data()
    vocab, vocab_to_idx, idx_to_vocab = process_data(text)
    encoded = np.array([vocab_to_idx[c] for c in text])

    # Train
    # train(vocab, encoded)

    # Predict
    samp = sample(vocab, vocab_to_idx, idx_to_vocab,
                  'checkpoint/i5000_l512.ckpt', 2000, lstm_size, len(vocab), prime='The')
    print(samp)


if __name__ == "__main__":
    main()
