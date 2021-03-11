# distutils: language=c++

import tensorflow as tf

cdef public void run():
    print("TF: Declaring vars...")
    a = tf.Variable(3.)
    x = tf.range(0, 100, 1.)
    y0 = tf.constant(2 * x + tf.random.uniform(x.shape, minval=-0.5, maxval=0.5))
    def forward(x, y0):
        y = a * x
        loss = tf.keras.losses.MSE(y, y0)
        return loss

    print("TF: Creating optimizers...")
    opt = tf.keras.optimizers.Adam(learning_rate=1e-1)

    print("TF: Running optimization...")
    for i_epoch in range(100):
        with tf.GradientTape() as g:
            grads = g.gradient(forward(x, y0), [a])
        opt.apply_gradients(zip(grads, [a]))
        print("{}: a = {}".format(i_epoch, a))

if __name__ == '__main__':
    run()
