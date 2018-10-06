"""
Implementation of importance sampling - evaluating the expected advantage of a policy 
using samples from an arbitrary policy.

@Authors: Avik Jain
"""
import numpy as np
import tensorflow as tf
from algorithms.architectures import feed_forward

def expected_advantage_importance_sampling(policy, sampled_policy, samples, discount_factor=1.0):
	"""
	Evaluates the expected advantage of policy using importance sampling from sampled_policy.

	Args:
		policy: PGFFNetwork representing policy to evaluate
		sampled_policy: PGFFNetwork representing policy that samples were drawn from
		samples: list of sample rollouts, sampled from sample_policy; each rollout is list 
				 of sample tuples (s, a, r)
	"""

	# helper function to take product of elements in list
	prod = lambda lst: reduce(lambda a, b: a*b, lst)

	expected_advantage = 0
	for rollout in samples:
		rollout_advantage = sum([r*(discount_factor**i) for i, (s, a, r) in rollout])

		# the following should include initial state probability and state transition probabilities, 
		# but they cancel when we divide
		policy_rollout_proba = prod([policy.action_dist(s)[a] for s, a, r in rollout])
		sampled_policy_rollout_proba = prod([sampled_policy.action_dist(s)[a] for s, a, r in rollout])

		scaled_advantage = policy_rollout_proba/sampled_policy_rollout_proba*rollout_advantage
		expected_advantage += scaled_advantage
	expected_advantage /= len(samples)
	return expected_advantage_importance_sampling

def off_policy_policy_gradient(policy, sampled_policy, samples, trainable_vars, sess):
	"""
	Evaluates the policy gradient with respect to the parameters of a policy, using samples drawn from a separate policy
	TODO: verify that this doesn't need to be built into a graph

	Args:
		policy: PGFFNetwork representing policy to evaluate
		sampled_policy: PGFFNetwork representing policy that samples were drawn from
		samples: list of sample rollouts, sampled from sample_policy; each rollout is list 
				 of sample tuples (s, a, r)
	"""
	result = np.zeros(trainable_vars.shape)
	for rollout_i, rollout in enumerate(samples):
		rollout_term = np.zeros(trainable_vars.shape)
		rollout_states = [s for s, a, r in rollout]

		remaining_rewards = sum([r for s, a, r in rollout])
		weight = 1
		for t, (s, a, r) in enumerate(rollout):
			timestep_term = sess.run(tf.gradients(tf.log(policy.output_tensor()[a]), trainable_vars), feed_dict={policy.state_tensor(): s})
			# update ratio of action probabilities for past actions
			weight *= policy.action_dist(s)[a]/sampled_policy.action_dist(s)[a]
			timestep_term *= weight # future actions don't affect current reward
			# update future rewards
			timestep_term *= remaining_rewards
			remaining_rewards -= r
			rollout_term += timestep_term
		result += rollout_term
	result /= len(samples)
	return result

