import gym
from gym import wrappers
import numpy as np
import random, tempfile, os
from collections import deque
import tensorflow.compat.v1 as tf
import tf_slim as slim
import matplotlib.pyplot as plt


class Main:
    def __init__(self, nS, nA, scope="estimator", learning_rate=0.0001, neural_architecture=None, global_step=None):
        self.nS = nS
        self.nA = nA
        self.global_step = global_step
        self.scope = scope
        self.learning_rate = learning_rate
        if not neural_architecture:
            neural_architecture = self.three_layers_network
        with tf.variable_scope(scope):
            self.create_network(network=neural_architecture, learning_rate=self.learning_rate)

    def three_layers_network(self,x,layer_1_nodes=32,layer_2_nodes=32,layer_3_nodes=32):
        layer_1=slim.fully_connected(x,layer_1_nodes,activation_fn=tf.nn.leaky_relu)
        layer_2=slim.fully_connected(layer_1,layer_2_nodes,activation_fn=tf.nn.leaky_relu)
        layer_3=slim.fully_connected(layer_2,layer_3_nodes,activation_fn=tf.nn.leaky_relu)
        return slim.fully_connected(layer_3,self.nA,activation_fn=None)

    def create_network(self,network,learning_rate=0.0001):
        tf.disable_eager_execution()
        self.X = tf.placeholder(shape=[None, self.nS], dtype=tf.float32, name="X")
        self.y = tf.placeholder(shape=[None, self.nA], dtype=tf.float32, name="y")
        self.predictions=network(self.X)
        sq_diff=tf.squared_difference(self.y,self.predictions)
        self.loss=tf.reduce_mean(sq_diff)
        self.train_op=slim.optimize_loss(self.loss,global_step=tf.train.get_global_step(),learning_rate=learning_rate,optimizer='Adam')

    def predict(self,sess,s): #prediction of q values for actions
        return sess.run(self.predictions,{self.X: s})

    def fit(self,sess,s,r,epochs=1):
        feed_dict={self.X: s,self.y: r}
        for epoch in range(epochs):
            res=sess.run([ self.train_op,self.loss,self.predictions],feed_dict)
            train_op,loss, predictions=res


class Memory:
    def __init__(self,memory_size=5000):
        self.memory=deque(maxlen=memory_size)

    def __len__(self):
        return len((self.memory))

    def add_memory(self,s,a,r,s_,status):
        self.memory.append((s,a,r,s_,status))

    def recall_memories(self):
        return list(self.memory)


class Agent:
    def __init__(self,nS,nA):
        #agent specifications
        self.nS=nS
        self.nA = nA
        self.epsilon=1.0
        self.epsilon_min=0.01
        self.epsilon_decay=0.999
        self.gamma=0.99
        self.learning_rate = 0.001
        self.epochs=1
        self.batch_size=30
        self.memory = Memory(memory_size=250000)
        self.global_step=tf.Variable(0,name='global_step',trainable=False)
        #for learning how to estimate the immediate next reward
        self.model = Main(nS=self.nS, nA=self.nA, scope="q",
                           learning_rate=self.learning_rate, global_step=self.global_step)
        # to store the weights for the final reward
        self.target_model = Main(nS=self.nS, nA=self.nA,scope="target_q",
                                  learning_rate=self.learning_rate,global_step=self.global_step)
        #to initialize the variables.
        init_op = tf.global_variables_initializer()
        #to save and restore all the variables.
        self.saver = tf.train.Saver()
        # Setting up the session
        self.sess = tf.Session()
        self.sess.run(init_op)

    def epsilon_update(self, t):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # ALlows the session to be saved
    def save_weights(self, filename):
        save_path = self.saver.save(self.sess,"%s.ckpt" % filename)
        print("Model saved in file: %s" % save_path)

    def load_weights(self, filename):
        self.saver.restore(self.sess, "%s.ckpt" % filename)
        print("Model restored from file")

    def set_weights(self, model_1, model_2):
        # to update the target Q network with the weights of the Q network
        # Enumerating and sorting the parameters
        # of the two models
        model_1_params = [t for t in tf.trainable_variables() \
                          if t.name.startswith(model_1.scope)]
        model_2_params = [t for t in tf.trainable_variables() \
                          if t.name.startswith(model_2.scope)]
        model_1_params = sorted(model_1_params,key=lambda x: x.name)
        model_2_params = sorted(model_2_params,key=lambda x: x.name)
        # Enumerating the operations to be done
        operations = [coef_2.assign(coef_1) for coef_1, coef_2 in zip(model_1_params, model_2_params)]
        # Executing the operations to be done
        self.sess.run(operations)

    def target_model_update(self):#Setting the model weights to the target model's ones
        self.set_weights(self.model, self.target_model)

    def act(self, s):

        # Based on epsilon predicting or randomly
        # choosing the next action
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.nA)
        else:
            # Estimating q for all possible actions
            q = self.model.predict(self.sess, s)[0]
            # Returning the best action
            best_action = np.argmax(q)
        return best_action

    def replay(self):
        batch = np.array(random.sample(self.memory.recall_memories(), self.batch_size))

        s = np.vstack(batch[:, 0])
        a = np.array(batch[:, 1], dtype=int)
        r = np.copy(batch[:, 2])
        s_p = np.vstack(batch[:, 3])
        status = np.where(batch[:, 4] == False)
        # We use the model to predict the rewards by
        # our model and the target model
        next_reward = self.model.predict(self.sess, s_p)
        final_reward = self.target_model.predict(self.sess, s_p)
        if len(status[0]) > 0:
            best_next_action = np.argmax(next_reward[status, :][0], axis=1)

            # adding the discounted final reward
            r[status] += np.multiply(self.gamma,final_reward[status, best_next_action][0])


        expected_reward = self.model.predict(self.sess,s)
        expected_reward[range(self.batch_size), a] = r

        self.model.fit(self.sess, s, expected_reward,epochs=self.epochs)

class Environment:
    def __init__(self, game="LunarLander-v2"):
        # Initializing
        np.set_printoptions(precision=2)
        self.env = gym.make(game)
        self.env = wrappers.Monitor(self.env, tempfile.mkdtemp(),force=True, video_callable=False)
        self.nS = self.env.observation_space.shape[0]
        self.nA = self.env.action_space.n
        self.agent = Agent(self.nS, self.nA)
        # Cumulative reward
        self.reward_avg = deque(maxlen=100)

    def test(self):
        return self.learn(epsilon=0.0, episodes=10,trainable=False, incremental=False)

    def train(self, epsilon=1.0, episodes=1500):
        return self.learn(epsilon=epsilon, episodes=episodes,trainable=True, incremental=False)

    def incremental(self, epsilon=0.01, episodes=100):
        return self.learn(epsilon=epsilon, episodes=episodes,trainable=True, incremental=True)

    def learn(self, epsilon=None, episodes=1000,trainable=True, incremental=False):

        if not trainable or (trainable and incremental):
            try:
                print("Loading weights")
                self.agent.load_weights('./weights.h5')
            except:
                print("Exception")
                trainable = True
                incremental = False
                epsilon = 1.0
        self.agent.epsilon = epsilon
        # Iterating through episodes
        summaries=[[],[],[]]
        for episode in range(episodes):
            episode_reward = 0
            s = self.env.reset()
            s = np.reshape(s, [1, self.nS])
            for time_frame in range(1000):
                if not trainable:
                    self.env.render()

                a = self.agent.act(s)
                s_p, r, status, info = self.env.step(a)
                s_p = np.reshape(s_p, [1, self.nS])

                episode_reward += r

                if trainable:
                    self.agent.memory.add_memory(s, a, r, s_p, status)

                s = s_p


                if trainable:
                    if len(self.agent.memory) > self.agent.batch_size:
                        self.agent.replay()
                # When the episode is completed,
                # exiting this loop
                if status:
                    if trainable:
                        self.agent.target_model_update()
                    break
            # Exploration vs exploitation
            self.agent.epsilon_update(episode)
            # Running an average of the past 100 episodes
            self.reward_avg.append(episode_reward)
            print("episode: %i score: %.2f avg_score: %.2f actions %i epsilon %.2f" % (episode,episode_reward, np.average(self.reward_avg), time_frame,self.agent.epsilon))

            summaries[0].append(episode)
            summaries[1].append(episode_reward)
            summaries[2].append(self.agent.epsilon)

        self.env.close()
        if trainable:
            self.agent.save_weights('./weights.h5')

        return summaries

def r_plot(summary):
    plt.plot(summary[0],summary[1], color='b', label='reward')
    plt.plot(summary[0],summary[2], color='r', label='epsilon')
    plt.xlabel("episodes")
    plt.ylabel("reward")
    plt.title("rewards for episodes")

lunar_lander = Environment(game="LunarLander-v2")

#summary1=lunar_lander.train()

#lunar_lander.incremental()
#r_plot(summary1)
summary2=lunar_lander.test()
r_plot(summary2)
plt.show()