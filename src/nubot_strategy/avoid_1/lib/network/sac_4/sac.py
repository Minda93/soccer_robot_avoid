import numpy as np
import tensorflow as tf

class ValueNetwork(object):
    def __init__(self, name):
        self.name = name

    def step(self, obs):
        with tf.variable_scope(self.name):
            """ state """
            pos = tf.slice(obs,[0,0],[-1,7])
            scan = tf.slice(obs,[0,7],[-1,-1])

            """ scan """
            scan_feature = tf.layers.dense(inputs = scan,\
                                           units = 512,\
                                           activation = tf.nn.relu6)
            scan_feature = tf.layers.dense(inputs = scan_feature,\
                                           units = 256,\
                                           activation = tf.nn.relu6)
            """ input """
            state_ = tf.concat([pos, scan_feature], axis=-1)

            """ layer """
            h1 = tf.layers.dense(inputs = state_,\
                                  units = 512,\
                                  activation = tf.nn.relu6)
            h2 = tf.layers.dense(inputs = h1,\
                                  units = 512,\
                                  activation = tf.nn.relu6)
            h2 = tf.layers.dense(inputs = h2,\
                                  units = 256,\
                                  activation = tf.nn.relu6)

            value = tf.layers.dense(h2, 1)
            value = tf.squeeze(value, axis=1)
            return value

    def get_value(self, obs):
        value = self.step(obs)
        return value


class QValueNetwork(object):
    def __init__(self, name):
        self.name = name

    def step(self, obs, action, reuse):
        with tf.variable_scope(self.name, reuse=reuse):
            """ state """
            pos = tf.slice(obs,[0,0],[-1,7])
            scan = tf.slice(obs,[0,7],[-1,-1])

            """ scan """
            scan_feature = tf.layers.dense(inputs = scan,\
                                           units = 512,\
                                           activation = tf.nn.relu6)
            scan_feature = tf.layers.dense(inputs = scan_feature,\
                                           units = 256,\
                                           activation = tf.nn.relu6)
            """ input """
            state_ = tf.concat([pos, scan_feature], axis=-1)
            state_m = tf.concat([state_, action], axis=-1)

            """ layer """
            h1 = tf.layers.dense(inputs = state_m,\
                                  units = 512,\
                                  activation = tf.nn.relu6)
            h2 = tf.layers.dense(inputs = h1,\
                                  units = 512,\
                                  activation = tf.nn.relu6)
            h2 = tf.layers.dense(inputs = h2,\
                                  units = 256,\
                                  activation = tf.nn.relu6)

            q_value = tf.layers.dense(h2, 1)
            q_value = tf.squeeze(q_value, axis=1)
            return q_value

    def get_q_value(self, obs, action, reuse=False):
        q_value = self.step(obs, action, reuse)
        return q_value


class ActorNetwork(object):
    def __init__(self,name,act_dim,action_bound): 
        self.act_dim = act_dim
        self.name = name
        self.action_bound = action_bound

        self.EPS = 1e-8

    def step(self, obs, log_std_min=-20, log_std_max=2):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            """ state """
            pos = tf.slice(obs,[0,0],[-1,7])
            scan = tf.slice(obs,[0,7],[-1,-1])

            """ scan """
            scan_feature = tf.layers.dense(inputs = scan,\
                                           units = 512,\
                                           activation = tf.nn.relu6)
            scan_feature = tf.layers.dense(inputs = scan_feature,\
                                           units = 256,\
                                           activation = tf.nn.relu6)
            """ input """
            state_ = tf.concat([pos, scan_feature], axis=-1)

            """ layer """
            h1 = tf.layers.dense(inputs = state_,\
                                  units = 512,\
                                  activation = tf.nn.relu6)
            h2 = tf.layers.dense(inputs = h1,\
                                  units = 512,\
                                  activation = tf.nn.relu6)
            h2 = tf.layers.dense(inputs = h2,\
                                  units = 256,\
                                  activation = tf.nn.relu6)

            mu = tf.layers.dense(h2, self.act_dim, None)
            log_std = tf.layers.dense(h2, self.act_dim, tf.tanh)
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

            std = tf.exp(log_std)
            pi = mu + tf.random_normal(tf.shape(mu)) * std

            pre_sum = -0.5 * (((pi - mu) / (tf.exp(log_std) + self.EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
            logp_pi = tf.reduce_sum(pre_sum, axis=1)

            mu = tf.tanh(mu)
            pi = tf.tanh(pi)

            clip_pi = 1 - tf.square(pi)
            clip_up = tf.cast(clip_pi > 1, tf.float32)
            clip_low = tf.cast(clip_pi < 0, tf.float32)
            clip_pi = clip_pi + tf.stop_gradient((1 - clip_pi) * clip_up + (0 - clip_pi) * clip_low)

            logp_pi -= tf.reduce_sum(tf.log(clip_pi + 1e-6), axis=1)
            return mu, pi, logp_pi

    def evaluate(self, obs):
        mu, pi, logp_pi = self.step(obs)
        
        mu *= self.action_bound
        pi *= self.action_bound

        return mu, pi, logp_pi


class SAC(object):
    def __init__(self, obs_dim,act_dim,action_bound,logpath):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.lr_actor = 0.001
        self.lr_value = 0.001
        self.gamma = 0.99
        self.tau = 0.995
        self.alpha = 0.2

        """ build model """
        self.OBS0 = tf.placeholder(tf.float32, [None, self.obs_dim], name="observations0")
        self.OBS1 = tf.placeholder(tf.float32, [None, self.obs_dim], name="observations1")
        self.ACT = tf.placeholder(tf.float32, [None, self.act_dim], name="action")
        self.RWD = tf.placeholder(tf.float32, [None,], name="reward")
        self.DONE = tf.placeholder(tf.float32, [None,], name="done")
        self.r_t = tf.placeholder(tf.float32, name="r_t")

        policy = ActorNetwork('Actor',act_dim,action_bound)
        q_value_net_1 = QValueNetwork('Q_value1')
        q_value_net_2 = QValueNetwork('Q_value2')
        value_net = ValueNetwork('Value')
        target_value_net = ValueNetwork('Target_Value')

        self.mu, self.pi, logp_pi = policy.evaluate(self.OBS0)

        q_value1 = q_value_net_1.get_q_value(self.OBS0, self.ACT)
        q_value1_pi = q_value_net_1.get_q_value(self.OBS0, self.pi, reuse=True)
        q_value2 = q_value_net_2.get_q_value(self.OBS0, self.ACT)
        q_value2_pi = q_value_net_2.get_q_value(self.OBS0, self.pi, reuse=True)
        value = value_net.get_value(self.OBS0)
        target_value = target_value_net.get_value(self.OBS1)

        min_q_value_pi = tf.minimum(q_value1_pi, q_value2_pi)
        next_q_value = tf.stop_gradient(self.RWD + self.gamma * (1 - self.DONE) * target_value)
        next_value = tf.stop_gradient(min_q_value_pi - self.alpha * logp_pi)

        policy_loss = tf.reduce_mean(self.alpha * logp_pi - q_value1_pi)
        q_value1_loss = tf.reduce_mean(tf.squared_difference(next_q_value, q_value1))
        q_value2_loss = tf.reduce_mean(tf.squared_difference(next_q_value, q_value2))
        value_loss = tf.reduce_mean(tf.squared_difference(next_value, value))
        total_value_loss = q_value1_loss + q_value2_loss + value_loss

        actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_actor)
        actor_train_op = actor_optimizer.minimize(policy_loss, var_list=tf.global_variables('Actor'))
        value_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_value)
        value_params = tf.global_variables('Q_value') + tf.global_variables('Value')

        with tf.control_dependencies([actor_train_op]):
            value_train_op = value_optimizer.minimize(total_value_loss, var_list=value_params)
        with tf.control_dependencies([value_train_op]):
            self.target_update = [tf.assign(tv, self.tau * tv + (1 - self.tau) * v)
                             for v, tv in zip(tf.global_variables('Value'), tf.global_variables('Target_Value'))]

        target_init = [tf.assign(tv, v)
                       for v, tv in zip(tf.global_variables('Value'), tf.global_variables('Target_Value'))]

        """ log """
        loss_actor = tf.summary.scalar("loss_actor",policy_loss)
        loss_q1 = tf.summary.scalar("loss_q1",q_value1_loss)
        loss_q2 = tf.summary.scalar("loss_q2",q_value2_loss)
        loss_value = tf.summary.scalar("loss_value",value_loss)
        loss_t_value = tf.summary.scalar("loss_t_value",total_value_loss)
        episode_reward = tf.summary.scalar("episode_reward",self.r_t)

        self.merge_summary = tf.summary.merge([loss_actor,\
                                               loss_q1,\
                                               loss_q2,\
                                               loss_value,\
                                               loss_t_value])
        self.reward_summary = tf.summary.merge([episode_reward])


        """ limit GPU"""
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(target_init)

        """ save """
        self.saver = tf.train.Saver()

        """ tensorflow """
        self.writer = tf.summary.FileWriter(logpath+"/TensorBoard/",graph = self.sess.graph)

    def Select_Action(self, obs, train):
        if(train):
            action = self.sess.run(self.pi, feed_dict={self.OBS0: obs.reshape(1, -1)})
            action = np.reshape(action,(self.act_dim,))
        else:
            action = self.sess.run(self.mu, feed_dict={self.state: state.reshape(1, -1)})
            action = np.reshape(action,(self.act_dim,))
        return action

    def Train(self,replay_buffer,iterations,episode = 0):
        if(episode == 0):
            obs0, act, rwd, obs1, done = replay_buffer.Sample(256)
            feed_dict = {self.OBS0: obs0,\
                        self.ACT: act,\
                        self.OBS1: obs1,\
                        self.RWD: rwd,\
                        self.DONE: np.float32(done)}
            self.sess.run(self.target_update, feed_dict)

            train_summary = self.sess.run(self.merge_summary, feed_dict)
            self.writer.add_summary(train_summary,iterations)
        else:
            for it in range(iterations):
                obs0, act, rwd, obs1, done = replay_buffer.Sample(256)
                feed_dict = {self.OBS0: obs0,\
                            self.ACT: act,\
                            self.OBS1: obs1,\
                            self.RWD: rwd,\
                            self.DONE: np.float32(done)}
                self.sess.run(self.target_update, feed_dict)
            
            train_summary = self.sess.run(self.merge_summary, feed_dict)
            self.writer.add_summary(train_summary,episode)

    def Log(self,reward,episode):
        feed_dict = {self.r_t: reward}
        reward_summary = self.sess.run(self.reward_summary, feed_dict)
        self.writer.add_summary(reward_summary,episode)
    
    def Save(self,directory,filename):
        path = "{}{}_Model.ckpt".format(directory,filename)
        self.saver.save(self.sess, path)
    
    def Load(self,directory,filename):
        checkpoint = tf.train.get_checkpoint_state("{}".format(directory))

        if(checkpoint and checkpoint.model_checkpoint_path):
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")