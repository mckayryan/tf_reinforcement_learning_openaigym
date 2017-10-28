import sys
import gym
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
NUM_EPISODES = 200  # Episode limitation
MAX_REWARD = 200
EP_MAX_STEPS = 200  # Step limitation in an episode
# The number of test iters (with epsilon set to 0) to run every TEST_FREQUENCY episodes
NUM_TEST_EPS = 5
HIDDEN_NODES = [24,16]
NETWORK = None

class net_config(object):
    iscontinuous = 0
    debug = 2
    initial_epsilon = 1
    epsilon_decay = 0.98
    learning_rate = 0.001
    double_q = 1
    dropout = 0
    hidden_nodes = [24,16]
    activation_fn = tf.nn.relu


class DQN(object):

    def __init__(self, state_dim, action_dim, load_net=False, **kwargs):
        self.config = net_config()
        self.epsilon = self.config.initial_epsilon
        self.p_net = self.t_net = self.variables = dict()

        if load_net == False:
            self.state_dim = state_dim
            self.action_dim = action_dim

            self.build_network()

            if self.config.double_q:
                self.build_t_network()
                printd("target network built\n{W}\n{b}\n{l}", W=self.t_net['W'], b=self.t_net['b'], l=self.t_net['L'])
                self.assign_primary_to_target()

            self.define_loss(self.select_action(self.p_net['Q'], self.variables['a_in']))
        else:
            self.load_network(kwargs)

    def build_network(self):
        self.s_in = tf.placeholder("float", [None, self.state_dim]),
        self.a_in = tf.placeholder("float", [None, self.action_dim]),
        self.t_in = tf.placeholder("float", [None])

        self.p_net['Q'], self.p_net['L'], self.p_net['W'], self.p_net['b'] = \
            self.define_q(self.variables['s_in'])

    def build_t_network(self):
        self.t_net['variables'] = {}
        self.t_s_in = tf.placeholder("float", [None, self.state_dim])

        self.t_net['Q'], self.t_net['L'], self.t_net['W'], self.t_net['b'] = \
            self.define_q(self.t_net['variables']['s_in'])

        self.t_q_idx = tf.placeholder('int32', [None, None], 't_q_idx')
        self.t_net['t_Q'] = \
            tf.gather_nd(self.t_net['Q'], self.t_net['variables']['q_idx'])

        self.t_net['variables']['t_w_input'] = {}
        self.t_net['variables']['t_w_assign_op'] = {}

    def assign_primary_to_target(self):

        for name in self.p_net['L'].keys():
            self.t_w_input = tf.placeholder('float32', self.t_net['W'][name].get_shape().as_list())

            printd('name {n}, t_net {t}, assigned {a}', level=1, n=name, \
                t=self.t_net['L'][name], a=self.t_net['variables']['t_w_input'][name])
            #self.t_net['variables']['t_w_assign_op'][name] = \
            #    tf.assign(self.t_net['W'][name], self.t_net['variables']['t_w_input'][name])

        self.update_t_net()

    def update_t_net(self):
        inputs = dict()
        for name in self.p_net['L'].keys():
            inputs[self.t_net['L'][name]] = self.p_net['L'][name]

        inputs[self.t_net['Q']] = self.p_net['Q']

        printd('\nInputs {tq}\n', tq=inputs)

        self.t_net['variables']['t_w_assign_op'] = \
            tf.contrib.graph_editor.graph_replace(self.t_net['Q'], inputs)

        printd('\nNew Target Q {tq}\n replaced to {assign}', tq=self.t_net['Q'], \
            assign=self.t_net['variables']['t_w_assign_op'])

    # def load_network(self, **kwargs):
    #     for key, value in kwargs.items():
    #         if key == 'Q':
    #             self.p_net['Q'] = value
    #         else:
    #             self.variables[key] = value

    def define_q(self, node_input):
        W = b = L = dict()
        for i, dim in enumerate(self.config.hidden_nodes):
            name = 'l'+str(i)
            L[name] =  tf.contrib.layers.fully_connected(node_input, dim, activation_fn=None)
            if self.config.dropout > 0:
                L[name] = tf.layers.dropout(L[name], rate=self.config.dropout)
            printd('{h} hidden nodes layer {n} assigned {s}: {ni}', h=dim, n=name, ni=L[name], s=L[name].shape)
            node_input = L[name]

        return tf.contrib.layers.fully_connected(node_input, self.action_dim, activation_fn=None), L, W, b

    def define_loss(self, action):
        self.variables['action'] = action
        self.variables['loss'] = tf.losses.mean_squared_error(self.variables['t_in'], tf.cast(action, dtype="float"))
        self.variables['optimizer'] = \
            tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.variables['loss'])
        self.variables['loss_op'] = train_loss_summary_op = tf.summary.scalar("TrainingLoss", self.variables['loss'])

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
        return self.variables['s_in'], self.variables['a_in'], self.variables['t_in'], self.p_net['Q'], \
            self.variables['action'], self.variables['loss'], self.variables['optimizer'], self.variables['loss_op']

    def eval_q(self, name='primary', inputs={}):
        if name == 'primary':
            q = self.p_net['Q']
        elif name == 'target':
            q = self.t_net['t_Q']
        return q.eval(inputs)



def printd(string, level=1, **kwargs):
    c = net_config()
    if c.debug >= level:
        print(string.format(**kwargs))

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
    global replay_buffer, epsilon
    replay_buffer = []
    epsilon = INITIAL_EPSILON

    state_dim = env.observation_space.shape[0]
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
    s_in = NETWORK.variables['s_in']
    printd("State {s} -> {d} -> {t} ", s=state, d=state.shape, t=state.dtype)
    Q_estimates = NETWORK.eval_q(name='primary',inputs={s_in: tf.transpose(state)})
    epsilon_to_use = 0.0 if test_mode else epsilon
    if random.random() < epsilon_to_use:
        action = random.randint(0, action_dim - 1)
    else:
        action = np.argmax(Q_estimates[0])
    printd("{q} -> {a}", level=1, q=Q_estimates, a=action)
    return action


def get_env_action(action):
    """
    Modify for continous action spaces that you have discretised, see hints in
    `init()`
    """
    return action


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


def get_target_q_value_batch(q_values, g_collection=tf.GraphKeys.TRAINABLE_VARIABLES):
    # establish exp moving average
    exp_average = tf.train.ExponentialMovingAverage(decay=0.999)
    # pull variables from
    variables = set( v.value() for v in tf.get_collection(g_collection))
    #print(variables)

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

    s_in = NETWORK.variables['s_in']
    q_value_batch = NETWORK.eval_q(name='primary', inputs={ s_in: next_state_batch})
    printd("Q Values {l}:\n{qv}", level=4, l=len(q_value_batch), qv=q_value_batch)

    if c.double_q:
        printd('Next Batch shape {s}\ts_in shape {ss}', s=next_state_batch.shape, ss=NETWORK.variables['s_in'].shape)
        predictions = np.argmax(NETWORK.eval_q(name='primary',inputs={s_in: next_state_batch}))

        t_s_in = NETWORK.t_net['variables']['s_in']
        t_q_idx = NETWORK.t_net['variables']['q_idx']

        printd("t_s_in {t}  +  t_q_idx {t_q}", t=t_s_in, t_q=t_q_idx)
        q_value_batch = NETWORK.eval_q(name='target', inputs={
            t_s_in: next_state_batch,
            t_q_idx: [[i,p] for i, p in enumerate(predictions)]
        })

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
        if test_mode: print("Test mode (epsilon set to 0.0)")
        ep_reward = 0
        for step in range(ep_max_steps):
            total_steps += 1

            # get an action and take a step in the environment
            action = get_action(state, None, NETWORK.p_net['Q'], NETWORK.epsilon, test_mode,
                                action_dim)

            env_action = get_env_action(action)
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
                do_train_step(replay_buffer, state_in, action_in, target_in,
                              q_values, q_selected_action, loss, optimise_step,
                              train_loss_summary_op, batch_presentations_count)
                batch_presentations_count += 1

            if done:
                break

        total_reward += ep_reward
        test_or_train = "test" if test_mode else "train"
        if test_mode:
            printd("end {tt} episode {ep}, reward: {r} - avg {ar:0.2f}, epsilon: {eps:.2}", level=1, \
            tt=test_or_train, ep=episode, r=ep_reward, ar=total_reward / (episode + 1), eps=epsilon)

        if all([ep_reward == MAX_REWARD, reached_max == 200]):
            reached_max = episode

    return { 'details':"G{g}Rs{rs}Hn{hn}".format(g=GAMMA, rs=REPLAY_SIZE, hn=HIDDEN_NODES), \
    'avgrew':total_reward / (episode + 1), \
    'maxrewreach': reached_max,
    'episodes':NUM_EPISODES}


def setup():
    default_env_name = 'CartPole-v0'
    #default_env_name = 'MountainCar-v0'
    # default_env_name = 'Pendulum-v0'
    # if env_name provided as cmd line arg, then use that
    env_name = sys.argv[1] if len(sys.argv) > 1 else default_env_name
    env = gym.make(env_name)
    state_dim, action_dim = init(env, env_name)
    network_vars = get_network(state_dim, action_dim)
    init_session()
    return env, state_dim, action_dim, network_vars


def main():

    result = []
    itr = 5
    for i in range(itr):
        env, state_dim, action_dim, network_vars = setup()
        result.append(qtrain(env, state_dim, action_dim, *network_vars, render=False))

    printd("{name}", level=0, name=result[0]['details'])
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
