import numpy as np
import gym
from gym import spaces

class barberisCasino(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is the environment described in [Barberis12] paper
    """

    metadata = {'render.modes': ['console']}

    def __init__(self, p = .5, bet = 10, T = 5):
        # calling gym.Env init
        super(barberisCasino, self).__init__() 
        
        # ==== Dynamics/Reward ====
        self.p = p # p (float): winning probability, < 1
        self.event = [1, 0, -1] # 1: wins, 0: draw, -1: lose
        self.Pmatrix = [self.p, 0, 1-self.p]
        self.bet = bet # bet (int): size of bets, assumed constant
        self.T = T # T (int): gambling horizon
        self.init_wealth = 0 # init_wealth (pos int): also used as CPT ref point
        
        # ==== Action ====
        self.action_space = spaces.Discrete(2) #0: "exit", 1: "cont"

        # ==== State ====
        self.observation_space = spaces.Tuple((spaces.Discrete(self.T + 1), spaces.Discrete(self.T * 2 + 1,)))
        #How to incl .bets value?
        
        #self.observation_space = spaces.Tuple((
        #    spaces.Box(low = 0 , high = 5 , shape = (self.T + 1,), dtype = np.int32),
        #    spaces.Box(low = -self.T * self.bet, high = self.T * self.bet, shape = (self.T * 2 + 1,), dtype = np.int32)))
        
        #self.observation_space = spaces.Tuple((
        #    spaces.Box(low = 0 , high = 5 , shape = (1,), dtype = np.int32),
        #    spaces.Box(low = -self.T * self.bet, high = self.T * self.bet, shape = (1,), dtype = np.int32)))
        
        #self.observation_space = spaces.Box(low = np.array([0, -self.T * self.bet]), high = np.array([5, self.T * self.bet]) , dtype = np.int32)
        
        # rarely used, but once DL is involved, it can be used to define matrices
        # we can use for tabular, too -- state_one_hot @value/policy_estimator
        
        self.time = None
        self.wealth = None    
        self.prev_time = None
        self.prev_wealth = None
    
    def seed(self):
        pass # passed cz deprecated; use env.reset(seed = seed) instead

    def reset(self, init_time = None, init_wealth = None, seed = None, return_info = False):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        if seed is not None and hasattr(super(), "seed"):
            super().seed(seed) # Alt: reset(seed = seed) --> ERROR NO ARGS SEED
        
        # @gym.Env <- super()
        # from gym.utils import seeding
        # if seed is not None:
        #    self._np_random, seed = seeding.np_random(seed)
        
        # case I: s0 = (0, 0)
        self.time = 0
        self.wealth = self.init_wealth
        
        # case II: self.time, self.wealth = self._sample_obs
        # may want to take care of wealth domain changes with t
        if init_time is not None and init_wealth is not None:
            self.time = init_time
            self.wealth = init_wealth
        
        return np.array([self.time, self.wealth], dtype = np.float32)
    
    #############################
    
    def step_(self, action, debug = False): #original, bf 'done' edits
        
        if action:  # continue: gamble and win (lose) w/ probability p (1-p)
                
            self.prev_time, self.prev_wealth = self._get_obs()
            
            self.wealth += self.bet * np.random.choice(self.event, 1, p = self.Pmatrix)[0]
            self.time += 1
            
            reward = self.wealth - self.prev_wealth
            
            if self.time >= self.T:
                done = True
            else: # cz at t = T-1, we still want to generate action -> apply step(act_T-1) return done = True, since otherwise we will generate a_T
                done = False
             
        else:  # exit: .. | this action should correspond to any t <= T-1, once "exit" we don't want to have act_t+1
            done = True
            
            self.prev_time, self.prev_wealth = self._get_obs()
            
            self.wealth += 0 # s' = s
            self.time += 1
            
            reward = self.wealth - self.prev_wealth # 0 reward
            
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        return self._get_obs(), reward, done, info
    
    ########################
    
    def step(self, action, debug = False):
        
        if action:  # continue: gamble and win (lose) w/ probability p (1-p)
            
            '''MOVE MARK LINES HERE???'''
            if self.time >= self.T:
                done = True
                
                self.prev_time, self.prev_wealth = self._get_obs()
                
                self.wealth += 0
                self.time += 1
                
                reward = self.wealth - self.prev_wealth
                
                
            else: # cz at t = T-1, we still want to generate action -> apply step(act_T-1) return done = True, since otherwise we will generate a_T
                done = False
                
                self.prev_time, self.prev_wealth = self._get_obs()
                
                self.wealth += self.bet * np.random.choice(self.event, 1, p = self.Pmatrix)[0]
                self.time += 1
                
                reward = self.wealth - self.prev_wealth
            
        else:  # exit: .. | this action should correspond to any t <= T-1, once "exit" we don't want to have act_t+1
            done = True
            
            self.prev_time, self.prev_wealth = self._get_obs()
            
            self.wealth += 0 # s' = s
            self.time += 1
            
            reward = self.wealth - self.prev_wealth # 0 reward
            
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        return self._get_obs(), reward, done, info
    
    def _get_obs(self):
        
        return self.time, self.wealth

    def _sample_obs(self): # take care of seeding
        
        t, idx = self.observation_space.sample() # idx: {0, 1, ..., 2*T}
        z = (idx - self.T) * self.bet # z: {-T*bet, ..., 0, ..., T*bet}
        
        return t, z
        
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

    def close(self):
        pass
    

