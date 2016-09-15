import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import pandas as pd

def load_data(dataset, nfold=5):
    print('... loading data')
    data = numpy.load(dataset)['data']

    numpy.random.seed(123)
    data = numpy.random.permutation(data)

    train_set = data[:-int(data.shape[0] / nfold)]
    test_set = data[-int(data.shape[0] / nfold):]

    def shared_dataset(data_xy, borrow=True):
        data_x = data_xy[:,:-1]
        data_y = data_xy[:,-1]
        shared_x = theano.shared(
                numpy.asarray(data_x, dtype=theano.config.floatX),
                borrow=borrow)
        shared_y = theano.shared(
                numpy.asarray(data_y, dtype=theano.config.floatX),
                borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
                value=numpy.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX),
                name='W', borrow=True)

        self.b = theano.shared(
                value=numpy.zeros(
                    (n_out,),
                    dtype=theano.config.floatX),
                name='b', borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        
        # parameters of the model
        self.params = [self.W, self.b]

class RBM(object):
    def __init__(self, input=None, n_visible=784, n_hidden=500,
        W=None, hbias=None, vbias=None, numpy_rng=None, theano_rng=None):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                    value=numpy.zeros(
                        n_hidden,
                        dtype=theano.config.floatX), 
                    name='hbias', borrow=True)

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                    value=numpy.zeros(
                        n_visible,
                        dtype=theano.config.floatX),
                    name='vbias', borrow=True)

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng

        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)

        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                n=1, p=h1_mean,
                dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)

        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                n=1, p=v1_mean,
                dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        ([
            pre_sigmoid_nvs,
            nv_means,
            nv_samples,
            pre_sigmoid_nhs,
            nh_means,
            nh_samples
        ], updates
        ) = theano.scan(
                self.gibbs_hvh,
                outputs_info=[None, None, None, None, None, chain_start],
                n_steps=k, name='gibbs_hbh')

        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        cross_entropy = T.mean(
                T.sum(
                    self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                    (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                    axis=1))
        return cross_entropy

def build_finetune_functions(self, datasets, batch_size, learning_rate):
    (train_set_x, train_set_y) = datasets[0]
    (test_set_x, test_set_y) = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_test_batches //= batch_size

    index = T.lscalar('index')  # index to a [mini]batch

    # compute the gradients with respect to the model parameters
    gparams = T.grad(self.finetune_cost, self.params)

    # compute list of fine-tuning updates
    updates = []
    for param, gparam in zip(self.params, gparams):
        updates.append((param, param - gparam * learning_rate))

    train_fn = theano.function(
        inputs=[index],
        outputs=self.finetune_cost,
        updates=updates,
        givens={
            self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
            self.y: train_set_y[index * batch_size: (index + 1) * batch_size]}
        )

    test_score_i = theano.function(
        [index],
        self.errors,
        givens={
            self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
            self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
            })

    # Create a function that scans the entire test set
    def test_score():
        return [test_score_i(i) for i in range(n_test_batches)]

    return train_fn, test_score

class DBN(object):
    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10):

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector
                                 # of [int] labels

        for i in range(self.n_layers):
            # construct the sigmoidal layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size, k):
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.In(learning_rate, value=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

def main(dataset,
        finetune_lr=0.1, pretraining_epochs=100,
        pretrain_lr=0.01, k=1, training_epochs=1000, batch_size=10,
        hidden_layers_sizes=[2000,2000,2000]):

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    n_in = train_set_x.shape[1].eval()
    n_out = len(set(train_set_y.eval()))

    # compute number of minibatches for training and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print('... building the model')
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=n_in,
              hidden_layers_sizes=hidden_layers_sizes,
              n_outs=n_out)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    print('... pre-training the model')
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    for i in range(dbn.n_layers):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost ' % (i, epoch), end=' ')
            print(numpy.mean(c))

    end_time = timeit.default_timer()
    print('The pretraining code for file ' + os.path.split(__file__)[1] 
            + ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training and testing function for the model
    print('... getting the finetuning functions')
    train_fn, test_model = build_finetune_functions(
            dbn,
            datasets=datasets, 
            batch_size=batch_size,
            learning_rate=finetune_lr
        )

    print('... finetuning the model')
    # early-stopping parameters
    patience = 20  # look as this many examples regardless

    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    test_frequency = min(n_train_batches, patience)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the test set; in this case we
                                  # check every epoch

    best_test_score = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    score = []
    print(patience, test_frequency)

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % test_frequency == 0:
                test_losses = test_model()
                this_test_score = numpy.mean(test_losses)
                print('epoch %i, minibatch %i/%i, test error %f %%' % 
                        (epoch, minibatch_index + 1, n_train_batches, this_test_score * 100.))

                score.append([epoch,this_test_score])

                # if we got the best test score until now
                if this_test_score < best_test_score:

                    #improve patience if loss improvement is good enough
                    if (this_test_score < best_test_score * improvement_threshold):
                        patience = patience + 10 * test_frequency

                    # save best test score and iteration number
                    best_test_score = this_test_score
                    best_iter = iter

            if patience * test_frequency <= iter:
                done_looping = True
                pass

    df = pd.DataFrame(score)
    spec = '%dx%d' % (hidden_layers_sizes[0], len(hidden_layers_sizes))

    if not os.path.exists('result'):
        os.makedirs('result')
    df.to_pickle('result/%s_%s.log' % (os.path.basename(dataset), spec))

    end_time = timeit.default_timer()

    print('Optimization complete with best test performance %f %% obtained at iteration %i' % 
            (best_test_score * 100., best_iter + 1))

    print('The fine tuning code for file ' + os.path.split(__file__)[1] + 
            ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)

if __name__ == '__main__':
    dataset = 'cpi.npz'
    if not os.path.exists(dataset):
        sys.exit('Please download cpi.npz from "https://my.syncplicity.com/share/vvks9oqxas1xneg/cpi"')
    main(dataset=dataset)
