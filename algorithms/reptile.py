import numpy as np
import tensorflow as tf
import random
import pickle
"""
Classes and functions for metalearning with repile and sampling from tasks. 
@Author Andrew Dickson
"""




class Reptile:
	def __init__(self, session, model, tasks, FLAGS, variables=None, load_buffer_from_pickle=False, save_buffer_to_pickle=False, load_buffer_pickle_path=None, save_buffer_pickle_path=None):
		self.sess = session
		self.model = model
		self.tasks = tasks
		self._model_state = VariableState(self.sess, variables or tf.trainable_variables())

		self.reward_threshold = float('inf') # TODO
		self.replay_buffer = {}
		self.save_buffer_to_pickle = save_buffer_to_pickle
		self.load_buffer_pickle_path = load_buffer_pickle_path
		self.save_buffer_pickle_path = save_buffer_pickle_path

		self.n_max_iter = FLAGS.max_episodes
		self.discount_rate = FLAGS.discount
		self.steps = FLAGS.steps
		self.single_task_steps = FLAGS.single_task_steps
		self.meta_step_size = FLAGS.meta_step_size
		self.env_act_n = FLAGS.env_act_n

		if load_buffer_from_pickle:
			try:
				with open(load_buffer_pickle_path, 'rb') as buffer_file:
					self.replay_buffer = pickle.load(buffer_file)
			except:
				self.replay_buffer = {}
		for task in tasks:
			self.replay_buffer[task] = []


		# Save model to be loaded later
		self.saver = tf.train.Saver(max_to_keep=2)

	def train(self):
		save_step = 20
		for step in range(self.steps):
			task_num, task = random.choice(list(enumerate(self.tasks)))
			#print(task, task_num)

			old_vars = self._model_state.export_variables()
			for i in range(self.single_task_steps):
				(states, actions, rewards, avg_len) = self.sample_from_task(self.model, task)
				feed_dict = {self.model.s: states, self.model.a: actions, self.model.r: rewards}
				self.sess.run(self.model.train_op, feed_dict=feed_dict)

				# compute discounted accumulated reward
				accumulated_reward = 0
				for inner_step in reversed(range(len(rewards))):
					accumulated_reward = rewards[inner_step] + accumulated_reward * self.discount_rate

				# Add rollouts with good performance to replay buffer for task for importance sampling
				if accumulated_reward > self.reward_threshold:
					self.replay_buffer[task].append((states, actions, rewards))
				print("Update: {}, Task: {}, Game Length: {}".format(i+step*self.single_task_steps,task_num, avg_len))

			new_vars = self._model_state.export_variables()

			if step % save_step == 0:
				print("SAVING MODEL", step)
				self.saver.save(self.sess, 'reptile_cart_model/meta-agent', global_step=step)

			self._model_state.import_variables(interpolate_vars(old_vars, new_vars, self.meta_step_size))

		if self.save_buffer_to_pickle:
			with open(self.save_buffer_pickle_path, 'wb') as buffer_file:
				pickle.dump(self.replay_buffer, buffer_file)


	def load(self):
		self.saver.restore(self.sess, self.saver.last_checkpoints[-1])
	#Sample a set of states, actions, and rewards from the model for a given task
	"""
	task: A game meta-learning will be used on. Must be openai environment. 
	model: tf model takes in state, outputs action distribution. Needs s and outputs. 
	"""
	def sample_from_task(self, model, task):
	
		# store states, actions, and rewards
		states = []
		actions = []
		rewards = []
		total_samples = 200

		games = 0
		while(len(states) < total_samples):
			games += 1
			obs = task.reset()
			for _ in range(self.n_max_iter):

				action_dist = self.sess.run(model.outputs, feed_dict={model.s: obs[np.newaxis, :]})
				action = np.random.choice(np.arange(self.env_act_n), p=np.squeeze(action_dist))
				new_obs, reward, done, info = task.step(action)

				states.append(obs)
				actions.append(action)
				rewards.append(reward)
				obs = new_obs
				if done:
					break

		# discount rewards
		discounted_rewards = []
		accumulated_reward = 0
		for step in reversed(range(len(rewards))):
			accumulated_reward = rewards[step] + accumulated_reward * self.discount_rate
			discounted_rewards.insert(0, accumulated_reward)
		# normalize discounted rewards
		discounted_rewards -= np.mean(discounted_rewards)
		discounted_rewards /= np.std(discounted_rewards)
		return (states, actions, discounted_rewards, len(states)/games)



class VariableState:
	"""
	Manage the state of a set of variables.
	"""
	def __init__(self, session, variables):
		self._session = session
		self._variables = variables
		self._placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape())
							  for v in variables]
		assigns = [tf.assign(v, p) for v, p in zip(self._variables, self._placeholders)]
		self._assign_op = tf.group(*assigns)

	def export_variables(self):
		"""
		Save the current variables.
		"""
		return self._session.run(self._variables)

	def import_variables(self, values):
		"""
		Restore the variables.
		"""
		self._session.run(self._assign_op, feed_dict=dict(zip(self._placeholders, values)))

def interpolate_vars(old_vars, new_vars, epsilon):
	"""
	Interpolate between two sequences of variables.
	"""
	return add_vars(old_vars, scale_vars(subtract_vars(new_vars, old_vars), epsilon))

def average_vars(var_seqs):
	"""
	Average a sequence of variable sequences.
	"""
	res = []
	for variables in zip(*var_seqs):
		res.append(np.mean(variables, axis=0))
	return res

def subtract_vars(var_seq_1, var_seq_2):
	"""
	Subtract one variable sequence from another.
	"""
	return [v1 - v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

def add_vars(var_seq_1, var_seq_2):
	"""
	Add two variable sequences.
	"""
	return [v1 + v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

def scale_vars(var_seq, scale):
	"""
	Scale a variable sequence.
	"""
	return [v * scale for v in var_seq]