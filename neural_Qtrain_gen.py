import sys
import gym
from gym.spaces import Discrete, Box
import tensorflow as tf
import numpy as np
import os
import random
import datetime
import json

"""
Hyper Parameters
"""
GAMMA = 0.95  # discount factor for target Q
INITIAL_EPSILON = 1  # starting value of epsilon
FINAL_EPSILON = 0.05  # final value of epsilon
EPSILON_DECAY_STEPS = 100
REPLAY_SIZE = 5000  # experience replay buffer size
BATCH_SIZE = 128  # size of minibatch
TEST_FREQUENCY = 20  # How many episodes to run before visualizing test accuracy
SAVE_FREQUENCY = 1000  # How many episodes to run before saving model (unused)
NUM_EPISODES = 100  # Episode limitation
MAX_REWARD = 1000
EP_MAX_STEPS = 1000  # Step limitation in an episode
# The number of test iters (with epsilon set to 0) to run every TEST_FREQUENCY episodes
NUM_TEST_EPS = 5
HIDDEN_NODES = [40,20]
NETWORK = None
is_continuous = False
action_map = dict()

import warnings
warnings.filterwarnings("ignore")

class net_config(object):
    iscontinuous = is_continuous
    debug = 0
    initial_epsilon = 1
    epsilon_decay = 0.98
    learning_rate = 0.001
    double_q = 1
    target_q_update_step = 20
    dropout = 1
    hidden_nodes = HIDDEN_NODES
    activation_fn = tf.nn.relu
    exp_moving_average = False


class DQN(object):

    def __init__(self, state_dim, action_dim, load_net=False, **kwargs):
        self.config = net_config()
        self.epsilon = self.config.initial_epsilon
        self.p_net = self.t_net = self.variables = dict()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.build_network()

        if self.config.double_q:
            self.build_t_network()
            printd("target network built\n{l}", level=3, l=self.t_net_layers)
            self.assign_primary_to_target()
            self.update_t_net()

        self.define_loss(self.select_action(self.q, self.a_in))

    def build_network(self):
        self.s_in = tf.placeholder("float", [None, self.state_dim])
        self.a_in = tf.placeholder("float", [None, self.action_dim])
        self.t_in = tf.placeholder("float", [None])
        printd("Network inputs established\ns: {s}\na: {a}\nt: {t}", level=3, s=self.s_in, a=self.a_in, t=self.t_in)
        self.q, self.p_net_layers = self.define_q(self.s_in)
        self.q_action = tf.argmax(self.q, axis=1)

        if self.config.exp_moving_average:
            self.ema_op = self.apply_ema()

        printd("Network q established {q} -> action {a}", level=3, q=self.q, a=self.q_action)

    def build_t_network(self):
        self.t_s_in = tf.placeholder("float", [None, self.state_dim])

        printd("t_s_in {d} {t}", level=3, d=self.t_s_in.shape, t=self.t_s_in)
        self.t_q, self.t_net_layers = self.define_q(self.t_s_in)

        self.t_q_idx = tf.placeholder('int32', [None, None], 't_q_idx')
        self.t_q_w_idx =  tf.gather_nd(self.t_q, self.t_q_idx)

    def assign_primary_to_target(self):
        self.t_w_input = self.t_w_assign_op = dict()

        for name in self.t_net_layers.keys():
            self.t_w_input[name] = tf.placeholder('float32', self.t_net_layers[name].get_shape().as_list())

            printd('name {n}, t_net {t}, assigned {a}', level=3, n=name, \
                t=self.t_net_layers[name], a=self.t_w_input[name])

    def update_t_net(self):
        inputs = dict()
        if self.config.exp_moving_average:
            for name in self.p_net_layers.keys():
                print(self.emas[name])
                inputs[self.emas[name]] = self.emas[name]
            inputs[self.emas['lq']] = self.emas['lq']
        else:
            for name in self.p_net_layers.keys():
                inputs[self.p_net_layers[name]] = self.p_net_layers[name]
            inputs[self.q] = self.q


        printd('\nInputs {tq}\n', level=4, tq=inputs)
        printd("Graph {g}", level=3, g=self.q)

        self.t_w_assign_op = tf.contrib.graph_editor.graph_replace(self.q, inputs)
        printd("Graph Replaced {g}", level=3, g=self.t_w_assign_op)
        printd('\nNew Target Q {tq}\n replaced to {assign}', level=4, tq=self.q, assign=self.t_w_assign_op)

    def define_q(self, node_input):
        L = dict()
        for i, dim in enumerate(self.config.hidden_nodes):
            name = 'l'+str(i)
            L[name] =  tf.contrib.layers.fully_connected(node_input, dim, activation_fn=tf.nn.relu)
            if self.config.dropout > 0:
                L[name] = tf.layers.dropout(L[name], rate=self.config.dropout)
            printd('{h} hidden nodes layer {n} assigned {s}: {ni}', level=4, h=dim, n=name, ni=L[name], s=L[name].shape)
            node_input = L[name]

        return (tf.contrib.layers.fully_connected(node_input, self.action_dim, activation_fn=None), L)

    def apply_ema(self):
        emas = dict()
        ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        # print([ self.p_net_layers[key] for key in self.p_net_layers.keys() ] + [self.q])
        ema.apply([ self.p_net_layers[key] for key in self.p_net_layers.keys() ] + [self.q])

        for name, layer in self.p_net_layers.keys():
            emas[name] = ema.average_name(name)
            print(emas[name])
        emas['lq'] = ema.average_name(self.q)

        return emas

    def clipped_error(self, value):
        if value >= -0.5 and value <= 0.5:
            f = value**2
        else:
            f = np.abs(value) - 0.25
        return f

    def define_loss(self, action):
        self.action = action
        printd("dim {t} x {a}", level=4, t=self.t_in.shape, a=self.action.shape)
        self.loss = tf.losses.mean_squared_error(self.t_in, tf.cast(self.action, dtype="float") )
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        self.loss_op = train_loss_summary_op = tf.summary.scalar("TrainingLoss", self.loss)

    def select_action(self, Q, a_in):
        return tf.reduce_sum(tf.multiply(Q, a_in), reduction_indices=1)

    def linear(self, X, dim, activation, name):
        in_shape = X.get_shape().as_list()
        W, b = tf.Variable(tf.truncated_normal([in_shape[-1]]+ [dim], stddev=0.1)), \
            tf.Variable(tf.constant(0.1, shape=[dim]))

        printd("{i} -> {ws}", level=1, i=in_shape, ws=W.get_shape().as_list())
        if len(in_shape) == 1:  X = tf.expand_dims(X,0)
        return tf.nn.relu(features=tf.matmul(X,W)+b), W, b

    def get_outputs(self):
        return self.s_in, self.a_in, self.t_in, self.q, \
            self.action, self.loss, self.optimizer, self.loss_op


def printd(string, level=1, **kwargs):
    c = net_config()
    if c.debug >= level:
        print(string.format(**kwargs))

def clamp_action(actions, low, high):
    return np.clip(actions, a_max=high, a_min=low)

def init(env, env_name):
    """
    Initialise any globals, e.g. the replay_buffer, epsilon, etc.
    return:
        state_dim: The length of the state vector for the env
        action_dim: The length of the action space, i.e. the number of actions

    NB: for discrete action envs such as the cartpole and mountain car, this
    function can be left unchanged.

    Hints for envs with continuous action spaces, e.g. "Pendulum-v0"
    1) you'll need to modify this function to discretise the action space and
    create a global dictionary mapping from action index to action (which you
    can use in `get_env_action()`)
    2) for Pendulum-v0 `env.action_space.low[0]` and `env.action_space.high[0]`
    are the limits of the action space.
    3) setting a global flag iscontinuous which you can use in `get_env_action()`
    might help in using the same code for discrete and (discretised) continuous
    action spaces
    """
    global replay_buffer, epsilon, is_continuous, action_map
    replay_buffer = []
    epsilon = INITIAL_EPSILON

    is_continuous = True if isinstance(env.action_space,Box) else False

    state_dim = env.observation_space.shape[0]

    if is_continuous:
        low, high = env.action_space.low[0], env.action_space.high[0]
        slices = action_dim = 21
        slice_unit = (high - low) / slices
        for s in range(slices):
            i = slices - s
            if (round((high - low) / i,5)*10).is_integer():
                action_dim = i
                slice_unit = (high - low) / i
                break

        for i in range(action_dim+1):
            action_map[i] = low + (i)*slice_unit

        #print(state_dim, action_dim, action_map)
    else:
        action_dim = env.action_space.n


    return state_dim, action_dim

def get_network(state_dim, action_dim, hidden_nodes=HIDDEN_NODES):
    """Define the neural network used to approximate the q-function

    The suggested structure is to have each output node represent a Q value for
    one action. e.g. for cartpole there will be two output nodes.

    Hints:
    1) Given how q-values are used within RL, is it necessary to have output
    activation functions?
    2) You will set `target_in` in `get_train_batch` further down. Probably best
    to implement that before implementing the loss (there are further hints there)
    """
    global NETWORK
    NETWORK = DQN(state_dim, action_dim)

    return NETWORK.get_outputs()

def init_session():
    global session, writer
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    # Setup Logging
    logdir = "tensorboard/" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, session.graph)

def get_action(state, state_in, q_values, epsilon, test_mode, action_dim):
    global NETWORK
    # printd("State {s} -> {d} -> {t} ", s=state, d=state.shape, t=state.dtype)
    epsilon_to_use = 0.0 if test_mode else epsilon
    if random.random() < epsilon_to_use:
        action = random.randint(0, action_dim - 1)
    else:
        action = NETWORK.q_action.eval(feed_dict={state_in: [state]})[-1]
    printd("{q} -> {a}", level=4, q=action, a=type(action))
    return action

def get_env_action(action):
    """
    Modify for continous action spaces that you have discretised, see hints in
    `init()`
    """
    global is_continuous, action_map
    if is_continuous:
        output_action = action_map[action]
    else:
        output_action = action
    return output_action

def update_replay_buffer(replay_buffer, state, action, reward, next_state, done,
                         action_dim):
    """
    Update the replay buffer with provided input in the form:
    (state, one_hot_action, reward, next_state, done)

    Hint: the minibatch passed to do_train_step is one entry (randomly sampled)
    from the replay_buffer
    """
    # TO IMPLEMENT: append to the replay_buffer
    # ensure the action is encoded one hot
    one_hot_action = np.zeros(action_dim)
    one_hot_action[action] = 1
    # append to buffer
    replay_buffer.append([state, one_hot_action, reward, next_state, done])
    # Ensure replay_buffer doesn't grow larger than REPLAY_SIZE
    if len(replay_buffer) > REPLAY_SIZE:
        replay_buffer.pop(0)
    return None

def do_train_step(replay_buffer, state_in, action_in, target_in,
                  q_values, q_selected_action, loss, optimise_step,
                  train_loss_summary_op, batch_presentations_count):
    target_batch, state_batch, action_batch = \
        get_train_batch(q_values, state_in, replay_buffer)
    printd("target_batch {s}", level=2, s=np.asarray(target_batch).shape)
    printd("state_batch {s}", level=2, s=np.asarray(state_batch).shape)
    printd("action_batch {s}", level=2, s=np.asarray(action_batch).shape)

    summary, _ = session.run([train_loss_summary_op, optimise_step], feed_dict={
        target_in: target_batch,
        state_in: state_batch,
        action_in: action_batch
    })
    writer.add_summary(summary, batch_presentations_count)

def get_train_batch(q_values, state_in, replay_buffer):
    """
    Generate Batch samples for training by sampling the replay buffer"
    Batches values are suggested to be the following;
        state_batch: Batch of state values
        action_batch: Batch of action values
        target_batch: Target batch for (s,a) pair i.e. one application
            of the bellman update rule.

    return:
        target_batch, state_batch, action_batch

    Hints:
    1) To calculate the target batch values, you will need to use the
    q_values for the next_state for each entry in the batch.
    2) The target value, combined with your loss defined in `get_network()` should
    reflect the equation in the middle of slide 12 of Deep RL 1 Lecture
    notes here: https://webcms3.cse.unsw.edu.au/COMP9444/17s2/resources/12494
    """
    global NETWORK
    c = net_config()
    minibatch = random.sample(replay_buffer, BATCH_SIZE)
    for i, item in enumerate(np.asarray(replay_buffer[:10])):
        printd("rb {l}-{s}:{eg}", level=3, l=i, s=len(item), eg=item)

    state_batch = np.asarray([data[0] for data in minibatch])
    action_batch = np.asarray([data[1] for data in minibatch])
    reward_batch = np.asarray([data[2] for data in minibatch])
    next_state_batch = np.asarray([data[3] for data in minibatch])

    printd("state_batch {l}:\n{eg}", level=3, l=len(state_batch), eg=state_batch[:10])
    printd("action_batch {l}:\n{eg}", level=3, l=len(action_batch), eg=action_batch[:10])
    printd("reward_batch {l}:\n{eg}", level=3, l=len(reward_batch), eg=reward_batch[:10])
    printd("next_state_batch {l}:\n{eg}", level=3, l=len(next_state_batch), eg=next_state_batch[:10])

    s_in = NETWORK.s_in
    q_value_batch = NETWORK.q.eval(feed_dict={s_in: next_state_batch})
    printd("Q Values {l}:\n{qv}", level=4, l=len(q_value_batch), qv=q_value_batch)

    if c.double_q:
        printd('Next Batch shape {s}\ts_in shape {ss}', level=3, s=next_state_batch.shape, ss=NETWORK.s_in.shape)
        predictions = NETWORK.q_action.eval(feed_dict={s_in: next_state_batch})
        printd("Predictions {s} {p}", level=3, s=predictions.shape, p=predictions[:10])
        t_s_in = NETWORK.t_s_in
        t_q_idx = NETWORK.t_q_idx

        printd("t_s_in {t}  +  t_q_idx {t_q}", level=3, t=t_s_in, t_q=t_q_idx)
        q_value_batch = NETWORK.t_w_assign_op.eval(feed_dict={
            s_in: next_state_batch
            # t_q_idx: [[i,p] for i, p in enumerate(predictions)]
        })
        printd("Before\n{b}\n+{c}", level=3, b=NETWORK.t_q.eval(feed_dict={t_s_in: next_state_batch}).shape, c=t_q_idx)
        printd("After\n{a}", level=3, a=q_value_batch.shape)

    target_batch = []

    for i in range(0, BATCH_SIZE):
        sample_is_done = minibatch[i][4]
        if sample_is_done:
            target_batch.append(reward_batch[i])
        else:
            # TO IMPLEMENT: set the target_val to the correct Q value update
            target_val = reward_batch[i] + GAMMA*np.amax(q_value_batch[i])
            target_batch.append(target_val)

    printd("targets {t}", level=4, t=target_batch[:10])

    return target_batch, state_batch, action_batch

def qtrain(env, state_dim, action_dim,
           state_in, action_in, target_in, q_values, q_selected_action,
           loss, optimise_step, train_loss_summary_op,
           num_episodes=NUM_EPISODES, ep_max_steps=EP_MAX_STEPS,
           test_frequency=TEST_FREQUENCY, num_test_eps=NUM_TEST_EPS,
           final_epsilon=FINAL_EPSILON, epsilon_decay_steps=EPSILON_DECAY_STEPS,
           force_test_mode=False, render=True):


    global epsilon
    global NETWORK

    # Record the number of times we do a training batch, take a step, and
    # the total_reward across all eps
    batch_presentations_count = total_steps = total_reward = 0

    config = net_config()

    reached_max = 200
    printd("Num Episodes {ne}", level=4, ne=num_episodes)

    for episode in range(num_episodes):
        # initialize task
        state = env.reset()
        if render: env.render()

        # Update epsilon once per episode - exp decaying
        #epsilon = INITIAL_EPSILON*np.exp(-episode/20)
        if NETWORK.epsilon > final_epsilon:
            NETWORK.epsilon *= config.epsilon_decay

        # in test mode we set epsilon to 0
        test_mode = force_test_mode or \
                    ((episode % test_frequency) < num_test_eps and
                        episode > num_test_eps
                    )
        #if test_mode: print("Test mode (epsilon set to 0.0)")
        ep_reward = 0
        for step in range(ep_max_steps):
            total_steps += 1

            # get an action and take a step in the environment
            action = get_action(state, NETWORK.s_in, NETWORK.q, NETWORK.epsilon, test_mode, action_dim)

            printd("action {a}", level=3, a=action)

            env_action = get_env_action(action)
            printd("env_action {a}", level=3, a=env_action)

            next_state, reward, done, _ = env.step(env_action)
            ep_reward += reward

            # display the updated environment
            if render: env.render()  # comment this line to possibly reduce training time

            # add the s,a,r,s' samples to the replay_buffer
            update_replay_buffer(replay_buffer, state, action, reward,
                                 next_state, done, action_dim)

            state = next_state
            printd("replay_buffer {l}, {lat}", level=4, l=len(replay_buffer), lat=replay_buffer[-1] )
            printd("Done {d}", level=4, d=done)
            # perform a training step if the replay_buffer has a batch worth of samples
            if (len(replay_buffer) > BATCH_SIZE):
                do_train_step(replay_buffer, NETWORK.s_in, NETWORK.a_in, NETWORK.t_in,
                              NETWORK.q, q_selected_action, loss, optimise_step,
                              train_loss_summary_op, batch_presentations_count)
                batch_presentations_count += 1

            if step % NETWORK.config.target_q_update_step == NETWORK.config.target_q_update_step - 1:
                NETWORK.update_t_net()

            if done:
                break

        total_reward += ep_reward
        test_or_train = "test" if test_mode else "train"
        if test_mode:
            printd("end {tt} episode {ep}, reward: {r} - avg {ar:0.2f}, epsilon: {eps:.2f}", level=1, \
            tt=test_or_train, ep=episode, r=ep_reward, ar=(total_reward / (episode + 1)), eps=NETWORK.epsilon)

        if all([ep_reward == MAX_REWARD, reached_max == 200]):
            reached_max = episode

    return { 'details':"G{g}Rs{rs}Hn{hn}".format(g=GAMMA, rs=REPLAY_SIZE, hn=HIDDEN_NODES), \
    'avgrew':total_reward / (episode + 1), \
    'maxrewreach': reached_max,
    'episodes':NUM_EPISODES}

def setup(env_in):
    default_env_name = 'CartPole-v0'
    #default_env_name = 'MountainCar-v0'
    # default_env_name = 'Pendulum-v0'
    # if env_name provided as cmd line arg, then use that
    env_name = sys.argv[1] if len(sys.argv) > 1 else env_in
    env = gym.make(env_name)
    state_dim, action_dim = init(env, env_name)
    network_vars = get_network(state_dim, action_dim)
    init_session()
    return env, state_dim, action_dim, network_vars


def main():
    global HIDDEN_NODES

    for env_name in ['CartPole-v0', 'MountainCar-v0', 'Pendulum-v0']:
        itr = 4
        result = []
        for i in range(itr):
            env, state_dim, action_dim, network_vars = setup(env_name)
            result.append(qtrain(env, state_dim, action_dim, *network_vars, render=False))

        printd("{name} - {env}", level=0, env=env_name, name=result[0]['details'])
        printd("max reward reached {rr} - average {avg}", level=0, rr=[ item['maxrewreach'] for item in result], \
        avg=np.sum([ item['maxrewreach'] for item in result])/itr)
        printd('Average Reward {ar}', level=0, ar=np.sum([ item['avgrew'] for item in result])/itr)
        printd('min/max Average Reward {minr}:{maxr}', level=0, minr=np.min([ item['avgrew'] for item in result]), \
        maxr=np.max([ item['avgrew'] for item in result]))
        printd('Total Episodes {te}', level=0, te=np.sum([ item['episodes'] for item in result]) )

    if not os.path.isfile("result.json"):
        with open('result.json', 'w') as fout:
            json.dump([result], fout)
    else:
        with open('result.json', 'r') as feedjson:
            feed = json.load(feedjson)
        feed.append(result)

        with open('result.json', 'w') as fout:
            json.dump(feed, fout)


if __name__ == "__main__":
    main()
