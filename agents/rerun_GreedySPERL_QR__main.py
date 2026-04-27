# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:38:31 2022

@author: NIXIESAP001
"""

#import gym
import itertools
#import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
import scipy.special as sp
import collections
#import random
#import math
import time
#import plotting
import record_csv
from datetime import datetime

import sys
import os

cwd = os.getcwd()
sys.path.append(cwd) # nb: append cwd or "../" not enough, need to add __init__ to the cwd;
if "../" not in sys.path:
    sys.path.append("../")
    
from lib.envs.barberis_casino import barberisCasino
from stable_baselines.common.misc_util import set_global_seeds

def barberisFeaturize(obs):
    t, z = obs
    obs_feature = np.zeros(env.observation_space.spaces[0].n * env.observation_space.spaces[1].n)
    loc = int(10 * t + 5 + z // env.bet)
    obs_feature[loc] = 1
    
    return obs_feature, loc

def zloc_to_zobs(zloc):
    
    return (zloc - 5)*env.bet



################################## POLICY #####################################

class GreedyPolicy():
    """
    Policy Function approximator | 
    @stable-baselines: .common.policies.PolicyType(sess, ob_space, ...)
    policy.value(obs, ...) returns Q(obs, self.action)
    """
    
    def __init__(self, env, critic_init, exploration, step_size = 1, THETA_MIN = .1, THETA_MAX = 1.0): 
        
        self.env = env
        self.name = 'GreedyPolicy'
        self.ss_const = step_size
        self.step_size = step_size
        
        self.explore = exploration['type'] # {softmax, eps-greedy} + hyperparameters
        self.eps = None
        self.alpha = None
        
        if self.explore == 'softmax':
            self.alpha = exploration['params'][0]
        else:
            self.eps = exploration['params'][0]
            
        self.nA = env.action_space.n
        
        self.THETA_MIN = THETA_MIN
        self.THETA_MAX = THETA_MAX
        
        self.theta = None
        self._get_theta(critic_init) 
        
        self.init_policy = None
        self.init_theta = np.copy(self.theta)
        
    def _get_theta(self, critic): # input: object
        
        self.theta = np.zeros((self.env.observation_space.spaces[0].n * self.env.observation_space.spaces[1].n, self.env.action_space.n))# vector of length d, init to 0^d
        
        for t in range(self.env.observation_space.spaces[0].n):
            for z in range(self.env.observation_space.spaces[1].n):
                state = t, zloc_to_zobs(z)
                _, loc = barberisFeaturize(state)
                critic_values = [critic.CPTpredict(state, a) for a in np.arange(self.nA)]
                
                # randomize argmax return, use:
                # np.random.choice(np.flatnonzero(critic_values == critic_values.max()))
                
                self.theta[loc, np.argmax(critic_values)] = 1
    
    def eps_greedy(self, arr):
        return arr * (1 - self.eps) + np.ones(self.nA, dtype=float) * (self.eps / self.nA)
    
    def softmax(self, arr):
        return sp.softmax(arr * self.alpha)
        
    def _clip(self, _theta): # clipping action?
        if _theta <= self.THETA_MIN:
            return self.THETA_MIN
        elif _theta >= self.THETA_MAX:
            return self.THETA_MAX
        else:
            return _theta        
    
    def _predict(self, state, deterministic = False):
        
        state, _ = barberisFeaturize(state) # dim = (1, 6 * 11)
        out = np.matmul(state, self.theta) # dim = (1, 2), val = [0, 1] or [1, 0]
        
        if deterministic:
            return out
        
        if self.explore == 'softmax':
            action_probs = self.softmax(out) # dim = (1, 2)
        else:
            action_probs = self.eps_greedy(out)
        
        return action_probs
    
    def _update(self, player_state, critic_values, tieBreak = 0, thresh = 0, tBRule = 'randomize'):
        # critic_values: Q(s, a) for all a
        
        if self.ss_const < 1e-16:
            return
        
        _, loc = barberisFeaturize(player_state)
        
        self.theta[loc, :] = np.zeros(self.nA)
        
        if tieBreak == 1:
            locmax = np.argmax(critic_values)
            # find all within thresh distance
            maxval = critic_values[locmax]
            loclist = [i for i, x in enumerate(critic_values) if abs(x-maxval) <= thresh]
            
            if tBRule == 'randomize':
                locmax, = np.random.choice(loclist, size=1)
            else:
                raise NotImplementedError
                #locmax = max(loclist) #min(loclist) #np.random.choice(loclist)
                #locmax = min(loclist)
                
            self.theta[loc, locmax] = 1
        else:
            self.theta[loc, np.argmax(critic_values)] = 1
        
##################################### CRITIC ##################################

class CPTCritic():
    
    def __init__(self, env, support_size = 10, with_quantile = False, with_huber = False, 
                 lbub = 0, treshRatio = 1000, param_init = 0.0, lr = .01):
        self.env = env
        self.with_quantile = with_quantile
        
        self.lr = lr
        self.cpt_theta = None
        self.with_huber = None
        self.support_size = None
        
        self.qtile_theta = None
        
        self.qtile_theta_pos = None
        self.qtile_theta_neg = None
        
        self.lossgrad_evolution = {}
        
        self.lbub = lbub
        self.treshRatio = treshRatio
        
        if not self.with_quantile:
            self.loss_grad = np.zeros((env.observation_space.spaces[0].n * env.observation_space.spaces[1].n, 
                                      env.action_space.n)) # (6*11, 2)
            self.cpt_theta = param_init + np.zeros(self.loss_grad.shape) # dim(cpt_theta) = dim(loss_grad) = 66, |A|
        else:
            self.with_huber = with_huber
            self.support_size = support_size # I
            self.loss_grad = np.zeros((env.observation_space.spaces[0].n * env.observation_space.spaces[1].n, 
                                      env.action_space.n, self.support_size)) # (6*11, 2, I)
            self.qtile_theta = param_init + np.zeros(self.loss_grad.shape) # dim(qtile_theta) = dim(loss_grad) = 66, |A|, I
            
            # assert qtile_theta.shape = qtile_theta_pos.shape = qtile_theta_neg.shape
            self.loss_grad_pos = np.zeros((env.observation_space.spaces[0].n * env.observation_space.spaces[1].n, 
                                      env.action_space.n, self.support_size))
            self.qtile_theta_pos = param_init + np.zeros(self.loss_grad.shape)
            self.loss_grad_neg = np.zeros((env.observation_space.spaces[0].n * env.observation_space.spaces[1].n, 
                                      env.action_space.n, self.support_size))
            self.qtile_theta_neg = param_init + np.zeros(self.loss_grad.shape)
            
    def cpt_featurize(self, state, action):
        featurized_sa = np.zeros(self.cpt_theta.shape) # dim = (66, |A|)
        _, loc = barberisFeaturize(state)
        featurized_sa[loc, action] = 1
        return featurized_sa
    
    def qtile_featurize(self, state, action, i):
        featurized_sai = np.zeros(self.qtile_theta.shape) # dim = (66, |A|, I)
        _, loc = barberisFeaturize(state)
        featurized_sai[loc, action , i] = 1
        return featurized_sai
    
    def CPTpredict(self, state, action, is_even = None): # cf. policy._predict()
        
        if not self.with_quantile:
            cpt_estimate = np.tensordot(self.cpt_featurize(state, action), self.cpt_theta, 
                                  axes = len(self.cpt_theta.shape)) # shape = (66, 2), num_axes = 2
            cpt_estimate = np.float(cpt_estimate)
            
        else:
            cpt_estimate = compute_CPT([self.qtilepredict(state, action, i, is_even) for i in range(self.support_size)], sort=False, lbub=self.lbub)
            
            ######################### IMPLEMENT SWITCH ########################
            cpt_ori = compute_CPT([self.qtilepredict(state, action, i, is_even) for i in range(self.support_size)], sort=False)
            #print(self.treshRatio)  # check default=np.inf
            if cpt_ori == 0 or abs((cpt_estimate - cpt_ori)/cpt_ori) > self.treshRatio:
                #print('HERE0, 1')
                #print('filtered:', cpt_estimate)
                #print('replaced w unfiltered:', cpt_ori)
                cpt_estimate = cpt_ori
            #else:
                #print('HERE2')
                #print('unfiltered:', cpt_ori)
                #print('use filtered:', cpt_estimate)
            ####################################################################
            
        return float(cpt_estimate)
    
    def qtilepredict(self, state, action, i, is_even = None): # by loss-grad @Dabney, use tabularFA, 
        # assert with_quantile == True;
        if is_even == None:
            qtile_estimate = np.tensordot(self.qtile_featurize(state, action, i), self.qtile_theta, 
                                          axes = len(self.qtile_theta.shape)) # shape = (66, 2, I), num_axes = 3
        elif is_even == False:
            #raise NotImplementedError
            qtile_estimate = np.tensordot(self.qtile_featurize(state, action, i), self.qtile_theta_pos, 
                                          axes = len(self.qtile_theta_pos.shape)) 
        else:
            #raise NotImplementedError
            qtile_estimate = np.tensordot(self.qtile_featurize(state, action, i), self.qtile_theta_neg, 
                                          axes = len(self.qtile_theta_neg.shape)) 
        
        return float(qtile_estimate)
    
    def compute_Loss(self, data):
        raise NotImplementedError
    
    def compute_LossGrad(self, data, is_even = None):
        ## manual backprop, given tabularFA of qtile_estimate;
        lossgrad = np.zeros(self.loss_grad.shape) # always clear grad
            
        if not self.with_quantile: # [MC-only]
            
            for state, action in data.keys(): # cf. CPT_val loop - @.evaluate_critic()
                # CHANGE CPT_dict.keys to ((., .), .)!!!
                
                cur_cpt_estimate = self.CPTpredict(state, action)
                cpt_target = compute_CPT(data[(state, action)])
                
                _, loc = barberisFeaturize(state)
                lossgrad[loc, action] += (cpt_target - cur_cpt_estimate)
                
        elif not self.with_huber: # [MC = TD?]
            for state, action in data.keys():
                qtile_targets = data[(state, action)]
                
                ############ FIRST VISIT: INIT TO MEAN TARGETS ################
                #WHAT IF AFTER MANY UPDATES all qtiles groundTruth are 0? 
                #For now, impossible by QRLoss.
                _, loc = barberisFeaturize(state)
                if sum(self.qtile_theta[loc, action, :] != np.zeros(self.support_size)) == 0:
                    '''print('first visit!', state, action)
                    print('bf:', self.qtile_theta[loc, action, :])
                    print('targets:', qtile_targets)'''
                    mean_targets = np.average(qtile_targets)
                    self.qtile_theta[loc, action, :] = [mean_targets for i in range(self.support_size)]
                    '''print('aft:', self.qtile_theta[loc, action, :])'''
                    
                # for t < 4, qtileTargets are TD targets --> they can be 0 at first visit!
                # The above holds even with backward updates, since OPPONENT_ACTION will determine the targets for t
                
                for i in range(self.support_size): # 1-1 <= i <= self.support_size-1
                    cur_qtile_estimate = self.qtilepredict(state, action, i, is_even)

                    tau_i = compute_cdf(i, self.support_size, pos = False) # = i/I
                    tau_i_next = compute_cdf(i+1, self.support_size, pos = False) # = (i+1)/I
                    midpoint_i = (tau_i + tau_i_next) / 2

                    _, loc = barberisFeaturize(state)
                    lossgrad[loc, action, i] += np.sum([midpoint_i - (z < cur_qtile_estimate) for z in qtile_targets])
                    # i: updates tau_i + tau_i+1/2; i=0 -> (0/50+1/50) / 2
                
                '''
                print('remove first visit!')
                print('bf:', self.qtile_theta[loc, action, :])
                print('then (pre *lr) lossGrad:', self.lr * lossgrad[loc, action, :]) # there is mid_point_i etc.
                '''
        else:
            raise NotImplementedError
        
        return lossgrad
    
    def _update(self, data, is_even = None):
        # [MC, w/ & w/o qtile]: call after loop 'm' (move out);
        # [TD, w/ qtile]: call inside loop 'm';
        # data (= CPTdict): pairs of predictors-targets, predictor: (s, a), target: Z^{k+1}(s, a); 
        
        if is_even == None:
            self.loss_grad = self.compute_LossGrad(data, is_even) 
            if not self.with_quantile: # [MC-only];    
                self.cpt_theta = self.cpt_theta + self.lr * self.loss_grad 
            else: # [MC, w/ qtile] = [TD, w/ qtile];
                # TD vs MC only affects 'data' s.t. for TD,
                # np.sum @.computelossgrad performs E_j[r + gamma * q(s',a',j)]
                self.qtile_theta = self.qtile_theta + self.lr * self.loss_grad
                                
        elif is_even == False:
            self.loss_grad_pos = self.compute_LossGrad(data, is_even)
            self.qtile_theta_pos = self.qtile_theta_pos + self.lr * self.loss_grad_pos
        else:
            self.loss_grad_neg = self.compute_LossGrad(data, is_even)
            self.qtile_theta_neg = self.qtile_theta_neg + self.lr * self.loss_grad_neg
  
################################### MAIN AGENT ################################

class QPG_CPT():
    """
    Prashant's CPT Policy Optimization Algorithm w/ SPSA Policy.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    def __init__(self, env, estimator_critic, estimator_policy, exploration,
                 seed = None, discount_factor = 1.0, target_type = None, order = None, 
                 state_only = False, support_size = 10, with_quantile = False, with_huber = False, 
                 lbub = 0, treshRatio = 1000,
                 param_init = 0.0, critic_lr = .1, empty_memory = True, 
                 step_size = 5.0, perturb_const = 1.9, actor_timescale = 1, 
                 use_buffer = 0, THETA_MIN = .1, THETA_MAX = 1.0):
        
        # env attr.
        self.env = env
        self.s0 = (0, env.init_wealth)
        self.horizon = self.env.T # self.T in barberisCasino
        
        self.seed = seed
        self.set_random_seed(seed = seed)
        
        # train (vf) attr.
        self.gamma = discount_factor
        self.CPT_dict = None
        self.ZList = None
        self.CPT_val = None
        self.CPT_true = None
        self.CPT_true_for_SPE = None
        
        self.vf_dict_init()
                
        self.empty_memory = empty_memory
        self.critic_class = estimator_critic
        self.critic = self.critic_class(self.env, support_size = support_size,
                                        with_quantile = with_quantile,
                                        with_huber = with_huber, lbub = lbub, treshRatio = treshRatio,
                                        param_init = param_init, lr = critic_lr)
        self.target_type = target_type
        self.order = order
        
        if order == 'fwd':
            self.player_set = range(self.horizon + 1)
        elif order == 'bwd':
            self.player_set = range(self.horizon, -1, -1) #reversed(range(self.horizon + 1))
        else:
            self.player_set = None
            
        self.state_only = state_only
        
        # train (policy) attr.
        self.policy_class = estimator_policy
        self.policy = self.policy_class(self.env, self.critic, exploration,
                                        step_size = step_size, THETA_MIN = THETA_MIN, THETA_MAX = THETA_MAX)
        self.actor_update_timescale = actor_timescale
        
        self.init_policy = None
        self.init_theta = None
        self.trained_policy = None
        self.trained_theta = None
        
        self.n_train_eps = None
        self.i_episode = None
        self.n_batch = None # batch size to obtain qtile (~vf) estimates
        self.use_buffer = use_buffer # alt: add .buffer_size
        self.logs = None # buffer logs containing Transitions
        
        # eval attr.
        self.eval_env = env # INCORPORATE ALGOEVAL HERE!!!
        self.n_eval_eps = None
        self.eval_freq = None
        self.best_mean_reward = -float("inf")
        self.verbose = None
        self.prev_hist_tuple = None
        self.prev_theta_histo = None
        
        self.stats = None
        self.SPSAwCritic_Check = None
        
        self.policy_dict_init()
        
    def vf_dict_init(self):
        self.CPT_dict = {}
        self.ZList = {} # w/o qtile, ZList = CPT_dict
        self.CPT_val = {}
        self.CPT_true = {}
        self.CPT_true_for_SPE = {}
        
        self.visitFreq = {}
        self.curVisitCounts = {}
        
        for t in range(self.env.observation_space.spaces[0].n):
            gain_supp = np.linspace(-self.env.bet * t, self.env.bet * t, t + 1)
            
            for id_ in range(self.env.observation_space.spaces[1].n):
                g = (id_ - self.env.T) * self.env.bet
                if g not in gain_supp:
                    continue
                
                state = (t, g)
                
                for action in range(self.env.action_space.n):
                    self.CPT_val[(state, action)] = []
                    self.CPT_true[(state, action)] = []
                    self.visitFreq[(state, action)] = []
                    self.curVisitCounts[(state, action)] = 0
        
    def policy_dict_init(self):
        self.stats = {'mean_rewards': [],
                      'std_rewards': [],
                      'cpt_rewards': []}
        self.SPSAwCritic_Check = {'QR': [], 'SQ': [], 'True': []}
        
    def set_random_seed(self, seed = None):
        
        if seed is None:
            return
        
        # Seed python, numpy and tf random generator
        set_global_seeds(seed) 
        
        if self.env is not None:
            self.env.reset(seed = seed) # set env seed
            self.env.action_space.seed(seed) # seed act_space: useful when selecting random actions
        # self.action_space.seed(seed)
    
    def _get_n_batch(self, n, const = 1):
        
        pass
        '''
        # to clarify: m_n = m_0 * num_iter**(-v) formula, (ii) m_n = trajectory len @Prash's code = 500 or sth else
        if self.policy.is_even:
            pass
        
        n_iterations = math.ceil(n / 2)
        self.n_batch = math.pow(n_iterations, -.1) * const # gamma = .101, alpha = 1 -> v < -.202'''
    
    def evaluate_policy(self, init_sa = None, deterministic = True, 
                        return_episode_rewards = True, for_critic = False):
        
        episode_rewards, episode_lengths = [], []
        
        for i in range(self.n_eval_eps):
            
            state = self.eval_env.reset()
            action = None
            init_t, init_g = state
            
            if init_sa is not None:
                state, action = init_sa
                init_t, init_g = state # 0, 0 if not init_sa?
                self.eval_env.reset(init_time = init_t, init_wealth = init_g)
            
            episode_reward = 0.0
            episode_length = 0
            
            for t in itertools.count(): 
                
                if t < init_t:
                    continue
                
                if t > init_t or action == None:
                    action_probs = self.policy._predict(state, deterministic)
                    '''print('@eval_policy, act_p:', action_probs)
                    print('for_critic', for_critic)
                    print('state:', state)'''
                    
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                    
                next_state, reward, done, _ = self.eval_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                    
                state = next_state
                
            episode_rewards.append(episode_reward + init_g)
            episode_lengths.append(episode_length)
        
        ## Policy Update Diagnostics
        if not for_critic:
            '''
            bins = np.linspace(-10, 10, num = 11) # {-10, -8, ..., 0, 2, .., 10}
            hist_tuple = np.histogram(np.array(episode_rewards), 
                                      bins = bins, density = True)
            
            plt.plot(hist_tuple[0], color = 'gray', 
                     alpha = .2 * self.i_episode/self.n_train_eps)
            print('---')
            '''
            if self.i_episode == 1:
                '''
                plt.plot(hist_tuple[0], color = 'k', label = 'theta0')
                '''
                init_policy = {}
                for t in range(self.eval_env.observation_space.spaces[0].n):
                    gain_supp = np.linspace(-self.eval_env.bet * t, self.eval_env.bet * t, t + 1)
                    
                    for id_ in range(self.eval_env.observation_space.spaces[1].n):
                        g = (id_ - self.eval_env.T) * self.eval_env.bet
                        
                        # remove unvisited states e.g. (0, -10), at which trained policies are irrelevant;
                        if g not in gain_supp:
                            continue
                        
                        state = (t, g)
                        action_probs = self.policy._predict(state, deterministic = True)
                        
                        if np.argmax(action_probs) == 1:
                            action = '~gamble'
                        else:
                            action = '~exit'
                        
                        if t == self.eval_env.T:
                            action = '~exit (default)'
                            
                        init_policy[state] = (list(action_probs), action)
                        
                print('Check init_policy:\n', init_policy) 
                self.init_policy = init_policy
                self.init_theta = self.policy.theta
                
            if self.i_episode == self.n_train_eps + 1:
                '''
                plt.plot(hist_tuple[0], color = 'r', label = 'theta*')
                '''
                
                final_policy = {}
                
                for t in range(self.eval_env.observation_space.spaces[0].n):
                    gain_supp = np.linspace(-self.eval_env.bet * t, self.eval_env.bet * t, t + 1)
                    
                    for id_ in range(self.eval_env.observation_space.spaces[1].n):
                        g = (id_ - self.eval_env.T) * self.eval_env.bet
                        
                        # remove unvisited states e.g. (0, -10), at which trained policies are irrelevant;
                        if g not in gain_supp:
                            continue
                        
                        state = (t, g)
                        action_probs = self.policy._predict(state, deterministic)
                        
                        if np.argmax(action_probs) == 1:
                            action = '~gamble'
                        else:
                            action = '~exit'
                        
                        if t == self.eval_env.T:
                            action = '~exit (default)'
                            
                        final_policy[state] = (list(action_probs), action)
                        
                print('Check final_policy:\n', final_policy) 
                self.trained_policy = final_policy
                self.trained_theta = self.policy.theta
            
        if return_episode_rewards:                                            
            return episode_rewards, episode_lengths
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        cpt_reward = compute_CPT(episode_rewards)
        
        if not for_critic:
            self.stats['mean_rewards'] += [mean_reward]
            self.stats['std_rewards'] += [std_reward]
            self.stats['cpt_rewards'] += [cpt_reward]
        
        return mean_reward, std_reward, cpt_reward #, true_qtiles
    
    def evaluate_critic_(self, is_even = None):
        
        for state, action in self.CPT_val.keys():
            self.CPT_val[(state, action)] += [[self.critic.CPTpredict(state, action, is_even)]]
            
            _, _, true_CPT = self.evaluate_policy(init_sa = (state, action), 
                                            deterministic = True, 
                                            return_episode_rewards = False,
                                            for_critic = True)
            self.CPT_true[(state, action)] += [[true_CPT]]
            self.visitFreq[(state, action)] += [[self.curVisitCounts[(state, action)]]]
            
            t, x = state
            if t < self.env.T:
                iterID = len(self.CPT_val[(state, action)]) - 1
                qtileVal = [self.critic.qtilepredict(state, action, i, is_even) for i in range(self.critic.support_size)]
                
                # filteredQtile, tresh = filtering(qtileVal, return_tresh = 1)
                # monoQtile, tresh_max = filtering(qtileVal, p_filter = 1, return_tresh = 1)
                # to_append = [[iterID] + qtileVal + [compute_CPT(qtileVal, sort=False)] + [tresh_max],
                #              [iterID] + monoQtile + [compute_CPT(monoQtile, sort=False)] + [tresh_max],
                #              [iterID] + filteredQtile + [compute_CPT(qtileVal, sort=False, lbub=self.critic.lbub)] + [tresh]] # CHECK LAST COLUMN EQUAL TO CPT_VALrow.csv
                
                to_append = [[iterID] + qtileVal + [compute_CPT(qtileVal, sort=False)],
                             [iterID] + filtering(qtileVal) + [compute_CPT(qtileVal, sort=False, lbub=self.critic.lbub)]] # CHECK LAST COLUMN EQUAL TO CPT_VALrow.csv
                
                saID = (state, action)
                record_csv.record_quantiles(to_append, envID, algoID, runID, saID)
                #to_append = [[iterID] + filtering(qtileVal) + [compute_CPT(qtileVal, sort=False, lbub=1)]]
                #record_csv.record_quantiles(to_append, envID, algoID, runID, saID, qtileTransform = 'lbub')
                '''
                print('t x a:', state, action)
                print('check cur cpt_val entry:', self.CPT_val[(state, action)])
                print('= lbub cpt?', compute_CPT(qtileVal, sort=False, lbub=self.critic.lbub))
                print('=old 130016 (seed12) cpt_val entry?')
                print('compare qtile_val, cpt no filter cur:', qtileVal, compute_CPT(qtileVal, sort=False))
                print('=old 130026?')'''
        
    def evaluate_critic(self, is_even = None):
        
        for state, action in self.CPT_dict.keys():
            if self.state_only: 
                # only use 1 action-id to keep data for C(s)
                action = 1
                
            XList = self.ZList[(state, action)]
            CPT_target = compute_CPT(XList)
            cur_cpt_est = self.critic.CPTpredict(state, action, is_even)
            
            if (state, action) not in self.CPT_val.keys():
                self.CPT_val[(state, action)] = [[cur_cpt_est, CPT_target, is_even]] # @bf_critic: CPT_target
            else:
                self.CPT_val[(state, action)] += [[cur_cpt_est, CPT_target, is_even]]
            
            if self.state_only:
                _, _, true_CPT = self.evaluate_policy(deterministic = True,
                                                      return_episode_rewards = False,
                                                      for_critic = True)
                '''_, _, true_CPT_explore = self.evaluate_policy(deterministic = False,
                                                                return_episode_rewards = False,
                                                                for_critic = True)'''
                true_CPT_explore = None
            else:
                _, _, true_CPT = self.evaluate_policy(init_sa = (state, action), 
                                                deterministic = True, 
                                                return_episode_rewards = False,
                                                for_critic = True)
                '''_, _, true_CPT_explore = self.evaluate_policy(init_sa = (state, action), 
                                                          deterministic = False,
                                                          return_episode_rewards = False,
                                                          for_critic = True)'''
                true_CPT_explore = None
                
            if (state, action) not in self.CPT_true.keys():
                self.CPT_true[(state, action)] = [[true_CPT, true_CPT_explore, is_even]]
            else:
                self.CPT_true[(state, action)] += [[true_CPT, true_CPT_explore, is_even]]
    
    
    def train_critic_TD(self, aggr_episodes, i_episode):
        
        # Iterate over ALL SAMPLED TRAJECTORIES
        for m in range(self.n_batch):
            accum_reward_ = np.sum([aggr_episodes[m][t].reward * self.gamma**t 
                                    for t in range(self.horizon + 1)
                                    if aggr_episodes[m][t] is not None])
            # Iterate over ALL PLAYERS (unordered: FWD, ordered: BWD/predict_order());
            # cf. MARL solver of simultaneous vs sequential games;
            
            for t in self.player_set:
                if aggr_episodes[m][t] == None: # for early done = True by act = 0;
                    continue
                
                state, action, reward, next_state, done = aggr_episodes[m][t]
                
                # Approximate Q-function or V-function of player t
                if done == True:
                    # Set terminal value for qtile/CPT estimation
                    init_t, init_g = state
                    accum_rewards_t = [init_g for j in range(self.critic.support_size)]
                    
                    # accum_rewards_t *= 1000 #resample_num
                    # do we need to adjust step_size here?
                    # ...
                    
                else:
                    # Greedy PI ~ (s, a, r, s', opponent's action a' ~ \pi'(.|s'))
                    opponent_act_probs = self.policy._predict(next_state, deterministic=True)
                    opponent_action = np.random.choice(np.arange(len(opponent_act_probs)), p=opponent_act_probs)
                    accum_rewards_t = [self.critic.qtilepredict(next_state, opponent_action, j) for j in range(self.critic.support_size)]
                    
                '''
                state = tuple(state)
                if (state, action) not in self.CPT_dict.keys():
                    self.CPT_dict[(state, action)] = accum_rewards_t
                    self.ZList[(state, action)] = [accum_reward_] # SQ estimates
                else:
                    self.CPT_dict[(state, action)] += accum_rewards_t
                    self.ZList[(state, action)] += [accum_reward_]
                '''
                # if t=T: update for all action, with QR (cz no assump on deterministic R(x_T, a_bar))
                if t == self.env.T:
                    for action in range(self.env.action_space.n):
                        if (state, action) not in self.CPT_dict.keys():
                            self.CPT_dict[(state, action)] = accum_rewards_t
                            self.ZList[(state, action)] = [accum_reward_] # SQ estimates
                        else:
                            self.CPT_dict[(state, action)] += accum_rewards_t
                            self.ZList[(state, action)] += [accum_reward_]
                            
                else:
                    state = tuple(state)
                    if (state, action) not in self.CPT_dict.keys():
                        self.CPT_dict[(state, action)] = accum_rewards_t
                        self.ZList[(state, action)] = [accum_reward_] # SQ estimates
                    else:
                        self.CPT_dict[(state, action)] += accum_rewards_t
                        self.ZList[(state, action)] += [accum_reward_]
                        
                # Update Q-/V-function Estimate of player t
                #print('Outside, t,x,a=', state, action)
                self.critic._update(data = self.CPT_dict)
                #print('check qtilePredict=', np.round([self.critic.qtilepredict(state, action, i) for i in range(self.critic.support_size)], 4))
                
                if self.empty_memory: # default = True
                    self.CPT_dict = {}
                    self.ZList = {}
                    
                # Update Best-Response Policy (direct from VF) of player t
                num_actions = self.env.action_space.n
                CPT_val_st = [self.critic.CPTpredict(state, a) for a in np.arange(num_actions)]
                
                # INCORPORATE TIE-BREAK THRESH HERE WITH INIT_SPERL TO {0, 1} RANDOM!!
                cur_a_max = np.argmax(CPT_val_st)
                prev_a_max = np.argmax(self.policy._predict(state, deterministic = True))
                if self.critic.CPTpredict(state, cur_a_max) > self.critic.CPTpredict(state, prev_a_max): 
                    self.policy._update(state, CPT_val_st)
                
                '''
                self.policy._update(state, CPT_val_st, tieBreak = 1, thresh = 0.3)
                '''
                
        pg_loss, vf_loss = None, None
        return pg_loss, vf_loss
    
    
    def train_critic_MC(self, aggr_episodes, i_episode):
        
        for m in range(self.n_batch):
            
            accum_reward_ = np.sum([aggr_episodes[m][t].reward * self.gamma**t 
                                    for t in range(self.horizon + 1)
                                    if aggr_episodes[m][t] is not None])
            
            # Iterate over ALL PLAYERS (unordered: FWD)
            for t in range(self.horizon + 1):
                
                if aggr_episodes[m][t] == None:
                    continue
                
                state, action, reward, next_state, done = aggr_episodes[m][t]
                
                # Approximate Q-function or V-function of player t
                accum_reward_t = accum_reward_ 
                
                state = tuple(state)
                if self.state_only:
                    action = 1
                
                if (state, action) not in self.CPT_dict.keys():
                    self.CPT_dict[(state, action)] = [accum_reward_t]
                    self.ZList[(state, action)] = [accum_reward_]
                else:
                    self.CPT_dict[(state, action)] += [accum_reward_t]
                    self.ZList[(state, action)] += [accum_reward_]
                
                # Update Q-/V-function Estimate of player t
                self.critic._update(data = self.CPT_dict)
                self.evaluate_critic()
                if self.empty_memory:
                    self.CPT_dict = {}
                    self.ZList = {}
                
                # Update Best-Response Policy (direct from VF) of player t
                num_actions = self.env.action_space.n
                CPT_val_st = [self.critic.CPTpredict(state, a) for a in np.arange(num_actions)]
                
                cur_a_max = np.argmax(CPT_val_st)
                prev_a_max = np.argmax(self.policy._predict(state, deterministic = True))
                if self.critic.CPTpredict(state, cur_a_max) > self.critic.CPTpredict(state, prev_a_max):
                    self.policy._update(state, CPT_val_st)
        
        pg_loss, vf_loss = None, None
        return pg_loss, vf_loss
    
    def assign_GainExit(self, n_eval_eps, init_sa = None):
        
        self.n_eval_eps = n_eval_eps
        self.policy.name = 'gainExit'
        
        analyt_precomm = {}
        
        for t in self.player_set:
            gain_supp = np.linspace(-self.eval_env.bet * t, self.eval_env.bet * t, t+1)
            for z in gain_supp:
                state = (t, z)
            
                if z > 0:
                    CPT_val_st = [1., 0.] # loss -> 'exit'
                else:
                    CPT_val_st = [0., 1.]
                
                self.policy._update(state, CPT_val_st)
                
                print('state:', state)
                print('policy, aft:', self.policy._predict(state, deterministic = True))
                
                analyt_precomm[state] = (self.policy._predict(state, deterministic = True), '')
                
        self.trained_policy = analyt_precomm
        
        _, _, true_CPT_s0 = self.evaluate_policy(deterministic = True,
                                                 return_episode_rewards = False,
                                                 for_critic = True)
        
        print('At p=', self.eval_env.p, ', Gain-exit Utility at s0:', true_CPT_s0)
        print('===')

    
    def assign_Precomm(self, n_eval_eps, init_sa = None):
        
        self.n_eval_eps = n_eval_eps
        self.policy.name = 'truePrecomm'
        
        analyt_precomm = {}
        
        for t in self.player_set:
            gain_supp = np.linspace(-self.eval_env.bet * t, self.eval_env.bet * t, t+1)
            for z in gain_supp:
                state = (t, z)
            
                if z < 0:
                    CPT_val_st = [1., 0.] # loss -> 'exit'
                else:
                    CPT_val_st = [0., 1.]
                
                self.policy._update(state, CPT_val_st)
                
                print('state:', state)
                print('policy, aft:', self.policy._predict(state, deterministic = True))
                
                analyt_precomm[state] = (self.policy._predict(state, deterministic = True), '')
                
        self.trained_policy = analyt_precomm
        
        _, _, true_CPT_s0 = self.evaluate_policy(deterministic = True,
                                                 return_episode_rewards = False,
                                                 for_critic = True)
        
        print('At p=', self.eval_env.p, ', Precomm(loss-exit) Utility at s0:', true_CPT_s0)
        print('===')
    
    def compute_SPE(self, n_eval_eps, init_sa = None, tieBreak = 0, thresh = 0):
        
        if self.order != 'bwd':
            print('Order must be bwd for compute_SPE!')
            raise ValueError
        
        self.n_eval_eps = n_eval_eps
        if tieBreak == 1:
            self.policy.name = 'SPE' + str(thresh) 
        else:
            self.policy.name = 'SPE' #'trueSPE'
            
        print('start compute SPE...')
        print('===')
        
        final_policy = {}
        
        for t in self.player_set:
            gain_supp = np.linspace(-self.eval_env.bet * t, self.eval_env.bet * t, t + 1)
            for z in gain_supp:
                state = (t, z)
                
                CPT_val_st = []
                for action in range(self.eval_env.action_space.n):
                    # compute_CPT given sample returns (n = n_eval_eps)
                    # default sort=True
                    _, _, true_CPT_sa = self.evaluate_policy(init_sa = (state, action), 
                                                          deterministic = True, 
                                                          return_episode_rewards = False, 
                                                          for_critic = True)
                    
                    if (state, action) not in self.CPT_true_for_SPE.keys():
                        self.CPT_true_for_SPE[(state, action)] = true_CPT_sa
                        CPT_val_st += [true_CPT_sa] 
                    else:
                        print('@compute_SPE, each (s, a) cannot be visited twice!')
                        print('s, a:', state, action)
                        raise ValueError
                
                print('state:', state)
                print('policy, bf:', self.policy._predict(state, deterministic = True))
                
                # IMPLEMENT TIE BREAK INSIDE THE FOLLOWING FUNCTION
                self.policy._update(state, CPT_val_st, tieBreak = tieBreak, thresh = thresh)
                
                print('C(s, a):', CPT_val_st)
                print('policy, aft:', self.policy._predict(state, deterministic = True))
                print('---')
        
                final_policy[state] = (self.policy._predict(state, deterministic = True), '')
        
        self.trained_policy = final_policy
        
        _, _, true_CPT_s0 = self.evaluate_policy(deterministic = True,
                                                 return_episode_rewards = False,
                                                 for_critic = True)
        print('At p=', self.eval_env.p, ', trueSPE Utility at s0:', true_CPT_s0)
        print('===')
                
    def learn(self, n_train_eps, init_sa = None, n_batch = 5, n_eval_eps = 50, eval_freq = 10, verbose = True):
        
        self.n_train_eps = n_train_eps
        self.n_batch = n_batch
        self.n_eval_eps = n_eval_eps
        self.eval_freq = eval_freq
        
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        
        for i_episode in range(1, n_train_eps + 1 + 1): # end+1 for eval records
            
            self.i_episode = i_episode
            
            if (i_episode - 1) % eval_freq == 0:
                start = time.time()
                episode_rewards, episode_lengths = self.evaluate_policy(init_sa = init_sa) # add init_sa for precomm-dyn;
                mean_reward, std_reward, cpt_reward = np.mean(episode_rewards), np.std(episode_rewards), compute_CPT(episode_rewards)
                mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
                
                self.stats['mean_rewards'] += [mean_reward]
                self.stats['std_rewards'] += [std_reward]
                self.stats['cpt_rewards'] += [cpt_reward]
                
                if verbose > 0:
                    print("\Eval @ Episode {}/{}, "
                          "episode_reward={:.2f} +/- {:.2f}".format(i_episode, n_train_eps, mean_reward, std_reward))
                    print("cpt_reward={:.2f}".format(cpt_reward))
                    print("episode_length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))
                    print("---")
                    
                if mean_reward > self.best_mean_reward:
                    if verbose > 0:
                        print('New best mean reward!')
                        print("---")
                    self.best_mean_reward = mean_reward
                print('EVALUATION time:', time.time() - start)
                
                self.evaluate_critic_()
                
            if (i_episode - 1) % self.n_batch == 0:
                aggr_episodes = np.array([[None for _ in range(self.horizon + 1)] for _ in range(self.n_batch)])
                m_id = 0
            
            state = self.env.reset() # add init_sa for precomm-dyn;
            action = None
            init_t, init_g = state
            
            if init_sa is not None:
                state, action = init_sa
                init_t, init_g = state # 0, 0 if not init_sa?
                self.env.reset(init_time = init_t, init_wealth = init_g)
            
            for t in itertools.count():
                
                if t < init_t:
                    continue
                
                if t > init_t or action == None:
                    action_probs = self.policy._predict(state)
                    #print(state, action_probs)
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                
                next_state, reward, done, _ = self.env.step(action)
                self.curVisitCounts[(tuple(state), action)] += 1
                
                aggr_episodes[m_id][t] = Transition(
                  state=state, action=action, reward=reward, next_state=next_state, done=done)
                
                if done:
                    break
                    
                state = next_state
            
            if m_id == self.n_batch - 1:
                
                #print('aggr_episodes:', aggr_episodes)
                
                start = time.time()
                if self.target_type == 'MC':
                    policy_loss, value_loss = self.train_critic_MC(aggr_episodes, i_episode)
                elif self.target_type == 'TD':
                    policy_loss, value_loss = self.train_critic_TD(aggr_episodes, i_episode)
                else:
                    raise NotImplementedError
                #print('TRAINING time, 1 batch:', time.time() - start)    
            
            m_id += 1  
            
            if self.logs is not None:
                self.logs = np.concatenate((self.logs, aggr_episodes), axis = 0)
            elif self.use_buffer:
                self.logs = aggr_episodes
            
        return self

#################################### RUNNER ###################################
#CPTParams = list(itertools.product(np.round(np.linspace(1, .95, 2), 4), np.round(np.linspace(1, .65+.05, 4), 4), np.round(np.linspace(1, 1.4, 3), 4)))
#CPTParams = [(.95, .5, 1.5), (.88, .65, 2.25)]

lbub = 1

# K = 50
#2I. reset CPTParams
CPTParams = [(.95, .5, 1.5)] #[(.88, .65, 2.25)] #[(.95, .5, 1.5)]
pwinArr = [.36, .3, .42] #[.62, .59, .64] # [.36, .3, .42]

# cpt=.95: .6, .63 wo filter/firstVisit alr decent (prob bias 0)
# cpt=1. : .63, .66, .72 alr decent
p_filterArr = [1.]
treshRatio = np.inf # equal to setting =0 (with MONO) if filter=1.
ss_inverted = 1

'''
#2II. reset CPTParams, w filters
CPTParams = [(.95, .5, 1.5)] # (1., 1., 1.), [~](.88, .65, 2.25)
pwinArr = [.72, .48, .66, .54, .6, .63] 
p_filterArr = [.75, .95, .8, .9, .85]
treshRatio = .5  # .25 may need for p.filter!=.75 | choose from range(.1, .9); ratio < 1 is needed to prevent crash to all 0s (sum(filter_) == 0);
ss_inverted = 1
'''

'''
# [DESKTOP] K=50, lr = 2.0
# [JS] K=200, lr={1.0, 2.0}
# [DESKTOP] K=50, lr=1.0
'''

'''
# SCHOOL
#1 II. Check: re-sweep other filters
CPTParams = [(.88, .65, 2.25)]
pwinArr = [ .72, .6, .54] #[.48, .54, .66*, .72, .6] # [.72], [.6] removed cz best is [1.]
p_filterArr = [.75, .95, .8, .9, .85] # [.75]
treshRatio = .5  # .7 may need for p.filter!=.75 | choose from range(.1, .9); ratio < 1 is needed to prevent crash to all 0s (sum(filter_) == 0);
# if lbub=1, default treshRatio = np.inf (all iters use cptFiltered)
ss_inverted = 1 # temp use for (1) FIRST VISIT (l. 267-275), (2) t == 5 CPT_dict (l.728-753)
'''

'''
#0. Check: BEST (pwin, pfilter) pairs for ss_inverted = 0, treshRatio = np.inf
# (.48 | 1., .75*, ) ~: running
# (.66 | 1., .75*, ) *: best, after firstVisit, bf re-sweep filter
# (.72 | 1.*, .75, ) .75 no longer best, cz 4201 bad samples > 3301 bias > SPE.5
# (.60 | 1.*, .75*, ) polErr n VErr different conclusion
# (.54 | 1., .75*, x.8^) ^: best, bf firstVisit; x: not yet 
CPTParams = [(.88, .65, 2.25)] #[(.95, .5, 1.5)]
pwinArr = [.72] # 
p_filterArr = [.75] # next: 4. reset cptParams w FILTER
treshRatio = .5 # .25 (p=.72 spikes but no effect on performance now, 
                #      will perform badly if spikes are down, n hits actGap,
                #      try bringing it down), 
                # np.inf (only if filter=1.; then, equal to setting =0 (with MONO))
ss_inverted = 1 # 0: only remove firstVisit, but keep the 2nd one; just want to test treshRatio for V00 fluctuate;
'''
seedArr = range(5, 14) #range(5, 14) # COMPLETE RUN: seed 5, for all

for alpha, rho1, lmbd in CPTParams:
    rho2 = rho1
    for p_filter in p_filterArr:
        
        def compute_cdf(k, q, pos):
            # compute cdf of k-th q quantile: 1 <= k <= q;
            
            if pos == True:
                return 1 - k/q
            else:
                return k/q
        
        def prob_weight(F, pos, rho1 = rho1, rho2 = rho2): 
            # @Prash: rho1 = .61, rho2 = .69 \in[0.3, 1]; assert w(F) monotonic in F;
            # @Barberis: rho1 = rho2 = .5 
        
            if pos == True:
                return F**rho1 / ((F**rho1 + (1-F)**rho1)**(1/rho1))
            else:
                return F**rho2 / ((F**rho2 + (1-F)**rho2)**(1/rho2))
        
        def utility(x, pos, alpha = alpha, lmbd = lmbd):
            # @Prash: alpha = .88 \in [0, 1], lmbd = 2.25 \in [1, 4];
            # @Barberis: alpha = .95, lmbd = 1.5 
            
            # compute u^{+/-}(x), where x is a quantile value: \in support(X);
            # X: accumulated reward r.v.;
            
            if pos == True:
                return x**alpha
            else:
                return -lmbd * (-x)**alpha 
        
        
        def filtering(quantiles, p_filter = p_filter):
            #print('INSIDE pfilter:', p_filter)
            #Insert discreteDistribFilter_
            qval_gaps = np.array(quantiles)[1:] - np.array(quantiles)[:-1] # len: K-1
            tresh_ = np.quantile(qval_gaps, p_filter, interpolation = 'higher') 
            #print('qval_gaps:', qval_gaps)
            #print('thresh_:', tresh_)
            
            filter_ = qval_gaps <= tresh_ + 1e-6 #NUMERICAL ERROR wo the addition!! | np.sum(filter_) ~36 (if K = 50)
            filter_ = np.multiply(filter_,  1 - np.multiply(qval_gaps < -1e-6, np.abs(qval_gaps - tresh_) > 1e-6))
            
            # if first elem of valGaps ok, then shd append [True]
            if filter_[0] != False:
                #print('HERE!!!')
                filter_ = np.append([True], filter_)
            else:
                filter_ = np.append([False], filter_) # before!!
            #print(sum(filter_), filter_)
             
            #Start filter quantiles: HALF-HALF REWRITE HERE
            # if True, copy quantiles
            qval_filtered = np.multiply(filter_, np.array(quantiles))
            #print(qval_filtered)
            #print(np.array(quantiles))
            
            ubVal = None
            qval_filtered = list(qval_filtered) + [np.inf]
            
            for i in range(len(qval_filtered)):
                
                if abs(qval_filtered[i]) >= 1e-6: # non-zero
                    ubVal = None # re-initialized...
                    continue
                
                #print('i:', i)
                #print('filtered:', qval_filtered)
                
                if i-1 < 0:
                    # update to ubVal
                    ubID = next(j for j, x in enumerate(qval_filtered) if j > i and abs(x) >= 1e-6)
                    if ubID > len(quantiles) - 1: # all 0!
                        qval_filtered[i] = 0.
                        continue
                    ubVal = quantiles[ubID]
                    
                    qval_filtered[i] = ubVal
                    
                else:
                    lbVal = qval_filtered[i-1]
                    
                    if ubVal is None:
                        ubID = next(j for j, x in enumerate(qval_filtered) if j > i and abs(x) >= 1e-6)
                        if ubID > len(quantiles) - 1:
                            qval_filtered[i] = lbVal
                            continue
                        ubVal = quantiles[ubID]
                    
                    distlb = quantiles[i] - lbVal
                    distub = ubVal - quantiles[i]
                    #print('qtileVal:', quantiles[i])
                    #print('dist lb, ub:', distlb, distub)
                    
                    # negative distlb!!! ONLY IF '0's leftmost!!!
                    # or if lb = ub
                    
                    if distlb < distub:
                        qval_filtered[i] = lbVal
                    else:
                        qval_filtered[i] = ubVal
            
            qval_filtered = qval_filtered[:-1]
            #print('qval_filtered:', qval_filtered)
            #print('cptVal:', compute_CPT(qval_filtered, sort=False))
            # can be due to init: 0 --> harder to learn 30 than 10!!
            
            return qval_filtered #, tresh_
        
        def compute_CPT(XList, sort = True, lbub = 0):
            if sort:
                XList = sorted(XList)
            
            # USE LBUBID TO INDICATE FILTERING, HERE WE MERGE BOTH
            if lbub == 1:
                XList = filtering(XList) # receive tresh_
                #XList = lbub_monotonic(XList)
                
            m = len(XList) 
            
            CPT_val_pos = 0
            CPT_val_neg = 0
            
            for id_, x in enumerate(XList):
                
                i = id_ + 1
                
                if x >= 0:
                    dF_i_pos = prob_weight(compute_cdf(i-1, m, True), True) - prob_weight(compute_cdf(i, m, True), True)
                    CPT_val_pos += utility(x, True) * dF_i_pos # CPT_val_neg += 0
                
                elif x < 0:
                    dF_i_neg = prob_weight(compute_cdf(i, m, False), False) - prob_weight(compute_cdf(i-1, m, False), False)
                    CPT_val_neg += utility(x, False) * dF_i_neg # CPT_val_pos += 0
                
                else:
                    print('@compute_CPT, x invalid!!!')
                    raise ValueError
                
            # ver: u^{-} non-decreasing, formula @compute_CPT() changed to C_n^{+} + C_n^{-}
            
            # if return_tresh == 1:
            # return CPT_val_pos + CPT_val_neg #, tresh_
            
            return CPT_val_pos + CPT_val_neg #, tresh_
        
        for p_win in pwinArr: # incl [.63, .6] #np.linspace(.69, .63, 4): # p = (.62, .65) breakpoint of 'gamble' at t0;
            p_win = np.round(p_win, 4)    
            for n_batch in [1]: # (default) [50]:
                for seed in seedArr:#range(5, 14): #[4]: #range(3, 6): #@p = .7: [18, 0]; @p = .5: [1, 0]; @p = .3: [4, 0]
                    for critic_lr in [2.0]: #[.2, .5]: # (default) [.1]:
                        step_size = 5.0 # 0.0 | 5.0
                        target_type = 'TD' # {'True', None, 'MC', 'TD{0/1}{0/1}'}
                        order = None
                        
                        # Ablation Parameters
                        support_size = 50 # check till (3, 30) bias disappears; but ideally need 200.
                        explore_type = 'eps-greedy' # 'softmax'
                        eps = .6 # default: eps=.3(<1), beta=10(>5)
                        if lbub == 0:
                            p_filter = 1 #NA
                            treshRatio = 0
                        if lbub == 1 and treshRatio == 0:
                            print('treshRatio = 0 will overwrite all filters with original quantiles!')
                            print('use lbub = 0 instead!')
                            raise ValueError
                        smoothen = 0
                        if treshRatio < np.inf:
                            smoothen = treshRatio 
                            #since we won't use smoothen in this script, use for treshRatio! default=0 means is not setup
                        
                        target_type_ = target_type
                        if target_type == 'TD':
                            critic_lr /= support_size
                            
                            TD_loc = True
                            Expected = False
                            order = 'bwd'
                            target_type_ = target_type_ + str(int(TD_loc)) + str(int(Expected)) + '(' + order + ')'
                            
                        #p_win = .7
                        env = barberisCasino(p = p_win) 
                        
                        actor_timescale = 1
                        train_num = 300*50 # change default, if n_batch is changed to 1 from 50; @BFS20: use 200*50
                        n_train_eps = train_num * (2*n_batch) * actor_timescale
                        n_eval_eps = 100   # S2 speedup: 500 -> 100 (5x per-eval); affects monitoring only
                        eval_freq = 2*n_batch *1500 # S1 speedup: factor 50 -> 1500 (15x fewer evals); was 100, now 3000
                        
                        # Default = (.1, 1.) -- changed for learning curve demo [cf. Debug0]
                        THETA_MIN = .1
                        THETA_MAX = 2.0
                        
                        hyperparams = [seed, p_win, alpha, rho1, rho2, lmbd, n_batch, train_num, actor_timescale, target_type_]
                        
                        print('CHOOSE ALGORITHM BLOCK(S) FROM THE FOLLOWING 4 OPTIONS: SPERL, SPE, PRECOMM, GAINEXIT')
                        print('--> To comment unnecessary blocks!')
                        
                        ##############################  SPERL #########################
                        ## 1a. Greedy PolImp
                        envID = 'barberis'
                        algoID = 'SPERL'
                        runID = datetime.now().strftime("%d%m%Y%H%M%S")
                        hyperparams_ = ['seed: ' + str(seed), 'p_win: ' + str(p_win),
                                       'alpha: ' + str(alpha), 'delta: ' + str((rho1, rho2)), 'lmbd: ' + str(lmbd), 'B: ' + str(0), 
                                       'critic_lr:' + str(critic_lr), 'target_type:' + str(target_type_), 'order:' + str(order), 
                                       'support_size(K) :' + str(support_size), 'lbub: ' + str(lbub), 'p_filter: ' + str(p_filter),
                                       'treshRatio: ' + str(treshRatio), 'smoothen: ' + str(smoothen), 'ss_inverted: ' + str(ss_inverted),
                                       'explore_type:' + str(explore_type), 'eps:' + str(eps), #'smoothen_thresh:' + str(smoothen_thresh),
                                       'n_batch:' + str(n_batch), #'by_episode:' + str(byEpisode),
                                       'train_num:' + str(train_num), 'n_eval:' + str(n_eval_eps)]
                        record_csv.record_params(hyperparams_, envID, algoID, runID)
                        
                        print('cpt:', [alpha, rho1, lmbd])
                        print('pwin:', p_win)
                        print('algo:', algoID)
                        print('filter:', p_filter)
                        print('treshRatio:', treshRatio)
                        print('firstVisit:', ss_inverted)
                        print('seed:', seed)
                        
                        model = QPG_CPT(env, estimator_critic = CPTCritic, estimator_policy = GreedyPolicy, 
                                        exploration = {'type': explore_type, 'params': [eps]},
                                        seed = seed, target_type = target_type, order = order, state_only = False,
                                        support_size = support_size, with_quantile = True, with_huber = False, lbub = lbub, treshRatio = treshRatio,
                                        param_init = 0.0, critic_lr = critic_lr, empty_memory = True, 
                                        step_size = step_size) #, perturb_const = perturb_const, actor_timescale = actor_timescale, THETA_MIN = THETA_MIN, THETA_MAX = THETA_MAX)
                        model.learn(n_train_eps = n_train_eps, n_batch = n_batch, 
                                    n_eval_eps = n_eval_eps, eval_freq = eval_freq)
                        
                        cptMdpParams = [alpha, rho1, lmbd, p_win]
                        ablationParams = [support_size, lbub*p_filter, smoothen, ss_inverted, eps]
                        record_csv.record_results(model, envID, algoID, runID, seed=seed, cptMdpID = cptMdpParams, ablationID = ablationParams)
                        
                        continue
                        
                        ##############################  SPE ###########################
                        ## 1c. vs True SPE
                        envID = 'barberis'
                        algoID = 'SPE'
                        runID = datetime.now().strftime("%d%m%Y%H%M%S")
                        
                        nEvalSPE = 2000
                        tieBreak_, thresh_ = 1, .4 # default: 0, 0
                        #tieBreak = 1, thresh = .5, .2, .1, .05   
                        
                        hyperparams_ = ['seed: ' + str(seed), 'p_win: ' + str(p_win),
                                       'alpha: ' + str(alpha), 'delta: ' + str((rho1, rho2)), 'lmbd: ' + str(lmbd), 'B: ' + str(0),
                                       'nEval: ' + str(nEvalSPE), 'tieBreak: ' + str(tieBreak_), 'thresh: ' + str(thresh_)]
                        record_csv.record_params(hyperparams_, envID, algoID, runID)
                        
                        print('algo:', algoID)
                        print('cpt:', [alpha, rho1, lmbd])
                        print('pwin:', p_win)
                        
                        print('n:', nEvalSPE)
                        print('tieBreak_:', tieBreak_)
                        print('thresh_:', thresh_)
                        print('seed:', seed)
                        
                        model_for_SPE = QPG_CPT(env, estimator_critic = CPTCritic, estimator_policy = GreedyPolicy, 
                                        exploration = {'type': explore_type, 'params': [eps]},
                                        seed = seed, target_type = target_type, order = order, state_only = False,
                                        support_size = support_size, with_quantile = True, with_huber = False, 
                                        param_init = 0.0, critic_lr = critic_lr, empty_memory = True, 
                                        step_size = step_size)
                        #model_for_SPE.compute_SPE(n_eval_eps = nEvalSPE)
                        model_for_SPE.compute_SPE(n_eval_eps = 2000, tieBreak = tieBreak_, thresh = thresh_)
                        
                        cptMdpParams = [alpha, rho1, lmbd, p_win]
                        record_csv.record_results(model_for_SPE, envID, algoID, runID, seed=seed, cptMdpID = cptMdpParams)
                        #record_csv.record_csv(model_for_SPE, hyperparams) # cannot implement tieBreak
                        
                        continue
                        
                        ############################  Precomm #########################
                        ## 1d. Loss-exit (~precomm) vs true SPE utility
                        # check if true_CPT(analyt_precomm) > SPSAtrained at p \in (.6, .65);
                        envID = 'barberis'
                        algoID = 'Precomm'
                        runID = datetime.now().strftime("%d%m%Y%H%M%S")
                        record_csv.record_params(hyperparams_, envID, algoID, runID)
                        
                        model_for_Precomm = QPG_CPT(env, estimator_critic = CPTCritic, estimator_policy = GreedyPolicy, 
                                                    exploration = {'type': explore_type, 'params': [eps]},
                                                    seed = seed, target_type = target_type, order = order, state_only = False,
                                                    support_size = support_size, with_quantile = True, with_huber = False, 
                                                    param_init = 0.0, critic_lr = critic_lr, empty_memory = True, 
                                                    step_size = step_size)
                        model_for_Precomm.assign_Precomm(n_eval_eps = 2000)
                        
                        record_csv.record_results(model_for_Precomm, envID, algoID, runID, seed=seed, cptMdpID = cptMdpParams)
                        record_csv.record_csv(model_for_Precomm, hyperparams)
                        
                        continue
                        
                        
                        ############################  GainExit #########################
                        ## 1d. Loss-exit (~precomm) vs true SPE utility
                        # check if true_CPT(analyt_precomm) > SPSAtrained at p \in (.6, .65);
                        envID = 'barberis'
                        algoID = 'GainExit'
                        runID = datetime.now().strftime("%d%m%Y%H%M%S")
                        
                        nEval = 2000
                        hyperparams_ = ['seed: ' + str(seed), 'p_win: ' + str(p_win),
                                       'alpha: ' + str(alpha), 'delta: ' + str((rho1, rho2)), 'lmbd: ' + str(lmbd), 'B: ' + str(0),
                                       'nEval: ' + str(nEval)]
                        record_csv.record_params(hyperparams_, envID, algoID, runID)
                        
                        model_for_GE = QPG_CPT(env, estimator_critic = CPTCritic, estimator_policy = GreedyPolicy, 
                                                    exploration = {'type': explore_type, 'params': [eps]},
                                                    seed = seed, target_type = target_type, order = order, state_only = False,
                                                    support_size = support_size, with_quantile = True, with_huber = False, 
                                                    param_init = 0.0, critic_lr = critic_lr, empty_memory = True, 
                                                    step_size = step_size)
                        model_for_GE.assign_GainExit(n_eval_eps = nEval)
                        
                        cptMdpParams = [alpha, rho1, lmbd, p_win]
                        record_csv.record_results(model_for_GE, envID, algoID, runID, seed=seed, cptMdpID = cptMdpParams)
                        record_csv.record_csv(model_for_GE, hyperparams)
                        
                        ######################### DYN-IMPLEMENT ###########################
                        '''
                        s0 = tuple(env.reset())
                        if s0 == (0, 0):
                            record_csv.init_csv(model, hyperparams)
                        record_csv.record_csv(model, hyperparams, model_for_SPE, model_for_Precomm)
                        #plotting.plot_results(model, hyperparams, model_for_SPE=model_for_SPE)
                        
                        #s0 = (0, 0)
                        dyn_env = barberisCasino(p = p_win) 
                        s0 = tuple(dyn_env.reset())
                        print('reset check, env.time:', dyn_env.time)
                        
                        model = QPG_CPT(env, estimator_critic = CPTCritic, estimator_policy = GreedyPolicy, 
                                        exploration = {'type': explore_type, 'params': [eps]},
                                        seed = seed, target_type = target_type, order = order, state_only = False,
                                        support_size = support_size, with_quantile = True, with_huber = False, 
                                        param_init = 0.0, critic_lr = critic_lr, empty_memory = True, 
                                        step_size = step_size)
                        #model_for_SPE = QPG_CPT()
                        #model_for_Precomm = QPG_CPT()
                        print('reset check, env.time:', dyn_env.time)
                        for t in itertools.count():
                            
                            print('@dyn-implement, s0:', s0)
                            model.learn(n_train_eps = n_train_eps, init_sa = (s0, None), n_batch = n_batch, 
                                        n_eval_eps = n_eval_eps, eval_freq = eval_freq)
                            #model_for_SPE.compute_SPE(n_eval_eps = 2000, init_sa = (s0, None))
                            #model_for_Precomm.assign_Precomm(n_eval_eps = 2000, init_sa = (s0, None))
                            print('reset check, env.time:', dyn_env.time)
                            
                            if s0 == (0, 0):
                                record_csv.init_csv(model, hyperparams, s0 = s0, dyn = True)
                            record_csv.record_csv(model, hyperparams, model_for_SPE, model_for_Precomm, s0 = s0, dyn = True)
                            
                            action_probs = model.policy._predict(s0, deterministic = True)
                            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                            print('a0, exit?', action)
                            
                            next_state, reward, done, _ = dyn_env.step(action)
                            
                            if done:
                                break
                            
                            s0 = next_state
                            
                            model.vf_dict_init()
                            model.policy_dict_init()
                            '''