#!/usr/bin/env python

import math
import numpy as np
import random
import sys
import tensorflow as tf

# See https://medium.com/@devnag/pointer-networks-in-tensorflow-with-sample-code-14645063f264

# Uncomment this to stop corner printing and see full/verbatim
#np.set_printoptions(threshold=np.nan)


def generate_nested_sequence(length, min_seglen=5, max_seglen=10):
    """Generate low-high-low sequence, with indexes of the first/last high/middle elements"""

    # Low (1-5) vs. High (6-10)
    seq_before = [(random.randint(1,5)) for x in xrange(random.randint(min_seglen, max_seglen))]
    seq_during = [(random.randint(6,10)) for x in xrange(random.randint(min_seglen, max_seglen))]
    seq_after = [random.randint(1,5) for x in xrange(random.randint(min_seglen, max_seglen))]
    seq = seq_before + seq_during + seq_after

    # Pad it up to max len with 0's
    seq = seq + ([0] * (length - len(seq)))
    return [seq, len(seq_before), len(seq_before) + len(seq_during)-1]


def create_one_hot(length, index):
    """Returns 1 at the index positions; can be scaled by client"""
    a = np.zeros([length])
    a[index] = 1.0
    return a


def get_lstm_state(cell):
    """Centralize definition of 'state', to swap .c and .h if desired"""
    return cell.c


def print_pointer(arr, first, second):
    """Pretty print the array, along with pointers to the first/second indices"""
    first_string = " ".join([(" " * (2 - len(str(x))) + str(x)) for x in arr])
    print(first_string)
    second_array = ["  "] * len(arr)
    second_array[first] = "^1"
    second_array[second] = "^2"
    if (first == second):
        second_array[first] = "^B"
    second_string = " " + " ".join([x for x in second_array])
    print(second_string)


def evaluate(max_length,         # J
             batch_size,         # B
             lstm_width,         # L
             num_blend_units,    # D
             num_training_loops,
             loss_interval,
             optimizer):
    """Core evaluation function given hyperparameters -- returns tuple of training losses and test percentage"""

    # S: Size of each vector (1 here, ignored/implicit)
    # I: num_indices (2 here; start and end)
    # J: input length (40 max here = max_length *), following notation of Vinyals (2015)
    # B: batch_size (param *)
    # L: lstm_width* units
    # D: Blend units

    num_indices = 2                         # I
    input_dimensions = 1                    # S  (dimensions per token)
    input_length = max_length               # J again
    generation_value = 20.0

    training_segment_lengths = (11, 20)     # Each of the low/high/low segment lengths
    testing_segment_lengths = (6, 10)       # "", but with no overlap whatsoever with the training seg lens

    reset_params = {"steps": 3000, "loss": .03}

    # Initialization parameters
    m = 0.0
    s = 0.5
    init = tf.random_normal_initializer(m, s)

    with tf.device("/cpu:0"):

        # Cleanup on aisle 6
        tf.reset_default_graph()

        # Training data placeholders
        inputs = tf.placeholder(tf.float32, name="ptr-in", shape=(batch_size, input_length))      # B x J
        # The one hot (over J) distributions, by batch and by index (start=1 and end=2)
        actual_index_dists = tf.placeholder(tf.float32,                                           # I x B x J
                                            name="ptr-out",
                                            shape=(num_indices, batch_size, input_length))

        # Define the type of recurrent cell to be used. Only used for sizing.
        cell_enc = tf.contrib.rnn.LSTMCell(lstm_width,
                                           input_size=None,
                                           use_peepholes=False,
                                           initializer=init)

        cell_dec = tf.contrib.rnn.LSTMCell(lstm_width,
                                           input_size=None,
                                           use_peepholes=False,
                                           initializer=init)

        # ###################  ENCODER
        enc_state = cell_enc.zero_state(batch_size, tf.float32)                # B x L: 0 is starting state for RNN
        enc_states = []
        with tf.variable_scope("rnn_encoder"):
            for j in xrange(max_length):
                if j > 0:
                    tf.get_variable_scope().reuse_variables()
                input_ = inputs[:, j:j+1]                                 # B x S : step through input, 1 batch at time

                # Map the raw input to the LSTM dimensions
                W_e = tf.get_variable("W_e", [input_dimensions, lstm_width], initializer=init)  # S x L
                b_e = tf.get_variable("b_e", [batch_size, lstm_width], initializer=init)        # B x L (bias matrix)
                cell_input = tf.nn.elu(tf.matmul(input_, W_e) + b_e)                            # B x L

                # enc state has c (B x L) and h (B x L)
                output, enc_state = cell_enc(cell_input, enc_state)

                enc_states.append(enc_state)   # c and h are each  B x L, and there will be J of them in list

        # ###################  DECODER
        # special symbol is max_length, which can never come from the actual data
        starting_generation_symbol = tf.constant(generation_value,                              # B x S
                                                 shape=(batch_size,
                                                        input_dimensions),
                                                 dtype=tf.float32)

        dec_state = enc_states[-1]  # final enc state, both c and h; they match as 2 ( B x L )
        ptr_outputs = []
        ptr_output_dists = []
        with tf.variable_scope("rnn_decoder"):
            input_ = starting_generation_symbol    # Always B x S

            # Push out each index
            for i in xrange(num_indices):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                # Map the raw input to the LSTM dimensions
                W_d_in = tf.get_variable("W_d_in", [input_dimensions, lstm_width], initializer=init)   # S x L
                b_d_in = tf.get_variable("b_d_in", [batch_size, lstm_width], initializer=init)         # B x L
                cell_input = tf.nn.elu(tf.matmul(input_, W_d_in) + b_d_in)                               # B x L

                output, dec_state = cell_dec(cell_input, dec_state)         # Output: B x L    Dec State.c = B x L

                # Enc/dec states (.c) are B x S
                # We want to map these to 1, right?  BxS and something that maps to B alone
                W_1 = tf.get_variable("W_1", [lstm_width, num_blend_units], initializer=init)            # L x D
                W_2 = tf.get_variable("W_2", [lstm_width, num_blend_units], initializer=init)            # L x D
                bias_ptr = tf.get_variable("bias_ptr", [batch_size, num_blend_units], initializer=init)  # B x D

                index_predists = []
                # Loop over each input slot to set up the softmax distribution
                dec_portion = tf.matmul(get_lstm_state(dec_state), W_2)                   # B x D

                enc_portions = []

                # Vector to blend
                v_blend = tf.get_variable("v_blend", [num_blend_units, 1], initializer=init)   # D x 1

                for input_length_index in xrange(input_length):
                    # Use the cell values (.c), not the output (.h) values of each state
                    # Each is B x 1, and there are J of them. Flatten to J x B
                    enc_portion = tf.matmul(get_lstm_state(enc_states[input_length_index]), W_1)         # B x D

                    raw_blend = tf.nn.elu(enc_portion + dec_portion + bias_ptr)                          # B x D
                    scaled_blend = tf.matmul(raw_blend, v_blend)                                         # B x 1
                    index_predist = tf.reshape(scaled_blend, (batch_size,))                              # B

                    enc_portions.append(enc_portion)
                    index_predists.append(index_predist)

                idx_predistribution = tf.transpose(tf.stack(index_predists))                             # B x J
                # Now, do softmax over predist, on final dim J (input length), to get to real dist
                idx_distribution = tf.nn.softmax(idx_predistribution, dim=-1)                            # B x J
                ptr_output_dists.append(idx_distribution)
                idx = tf.argmax(idx_distribution, 1)  # over last dim, rank reduc                        # B

                # Pull out the input from that index
                emb = tf.nn.embedding_lookup(tf.transpose(inputs), idx)                                  # B x B
                ptr_output_raw = tf.diag_part(emb)                                                       # B

                ptr_output = tf.reshape(ptr_output_raw, (batch_size, input_dimensions))                  # B x S
                ptr_outputs.append(ptr_output)
                input_ = ptr_output    # The output goes straight back in as next input

        # Compare the one-hot distribution (actuals) vs. the softmax distribution: I x (B x J)
        idx_distributions = tf.stack(ptr_output_dists)                                                   # I x B x J

        # ############## LOSS
        # RMS of difference across all batches, all indices
        loss = tf.sqrt(tf.reduce_mean(tf.pow(idx_distributions - actual_index_dists, 2.0)))
        train = optimizer.minimize(loss)

        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)

        # ############## TRAINING
        train_dict = {}
        sequences = []
        first_indexes = []
        second_indexes = []

        # Note that our training/testing datasets are the same size as our batch. This is
        #   unusual and just makes the code slightly simpler. In general your dataset size
        #   is >> your batch size and you rotate batches from the dataset through.
        for batch_index in xrange(batch_size):
            data = generate_nested_sequence(max_length,
                                            training_segment_lengths[0],
                                            training_segment_lengths[1])
            sequences.append(data[0])                                           # J
            first_indexes.append(create_one_hot(input_length, data[1]))         # J
            second_indexes.append(create_one_hot(input_length, data[2]))        # J

        train_dict[inputs] = np.stack(sequences)                                # B x J
        train_dict[actual_index_dists] = np.stack([np.stack(first_indexes),     # I x B x J
                                          np.stack(second_indexes)])

        losses = []
        for step in xrange(num_training_loops):
            tf_outputs = [loss, train, idx_distributions, actual_index_dists]
            results = sess.run(tf_outputs, feed_dict=train_dict)
            step_loss = results[0]

            if step % loss_interval == 0:
                losses.append(step_loss)
                print("%s: %s" % (step, step_loss))
                sys.stdout.flush()
            if step >= reset_params["steps"] and step_loss > reset_params["loss"]:
                return None

        # ############## TESTING
        print(" === TEST === ")

        sequences = []
        first_indexes = []
        second_indexes = []
        for batch_index in xrange(batch_size):
            data = generate_nested_sequence(max_length,
                                            testing_segment_lengths[0],
                                            testing_segment_lengths[1])
            sequences.append(data[0])                                           # J
            first_indexes.append(create_one_hot(input_length, data[1]))         # J
            second_indexes.append(create_one_hot(input_length, data[2]))        # J

        test_dict = {inputs: np.stack(sequences),
                     actual_index_dists: np.stack([np.stack(first_indexes),
                                                   np.stack(second_indexes)])}
        # 0 is loss, 1 is prob dists, 2 is actual one-hots
        results = sess.run([loss, idx_distributions, actual_index_dists], feed_dict=test_dict)
        print("Test %s: loss %s" % (i, results[0]))

        incorrect_pointers = 0
        for batch_index in xrange(batch_size):

            first_diff = first_indexes[batch_index] - results[1][0][batch_index]
            first_diff_max = np.max(np.abs(first_diff))
            first_ptr = np.argmax(results[1][0][batch_index])
            if first_diff_max >= .5:  # bit stricter than argmax but let's hold ourselves to high standards, people
                incorrect_pointers += 1
            second_diff = second_indexes[batch_index] - results[1][1][batch_index]
            second_diff_max = np.max(np.abs(second_diff))
            second_ptr = np.argmax(results[1][1][batch_index])
            if second_diff_max >= .5:
                incorrect_pointers += 1

            print_pointer(sequences[batch_index], first_ptr, second_ptr)
            print("")

        test_pct = np.round(100.0 * ((2 * batch_size) - incorrect_pointers) / (2 * batch_size), 5)
        print("")
        print(" %s / %s (correct/total); test pct %s" % ((2*batch_size) - incorrect_pointers,
                                                         2 * batch_size,
                                                         test_pct))
        sys.stdout.flush()

    return losses, test_pct


max_reset_retries = 20
for reset_loop_index in xrange(max_reset_retries):

    # Create optimizer - AdaGrad works well on this problem
    learning_rate = 1.0
    adagrad_optimizer = tf.train.AdagradOptimizer(learning_rate)

    lstm_blend = 6
    result = evaluate(max_length=60,
                      batch_size=1024,
                      lstm_width=lstm_blend,
                      num_blend_units=lstm_blend,
                      num_training_loops=4000,
                      loss_interval=50,
                      optimizer=adagrad_optimizer)

    if result is None:
        print("Warning: loss is stagnant-- starting again")
    else:
        print("Training losses: %s" % (str(result[0])))
        print("Test percentage: %s" % (result[1]))
        break  # We're done!
