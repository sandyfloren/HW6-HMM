import numpy as np


class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(
            self, 
            observation_states: np.ndarray, 
            hidden_states: np.ndarray, 
            prior_p: np.ndarray, 
            transition_p: np.ndarray, 
            emission_p: np.ndarray
            ):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p = prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p

        self.n_states = len(hidden_states)


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        input_observation_states = np.atleast_1d(input_observation_states)

        n_obs = input_observation_states.size
        if n_obs == 0: 
            raise ValueError('Input sequence is empty.')
        for state in np.unique(input_observation_states):
            if state not in self.observation_states:
                raise ValueError(f'Invalid state: {state}')
        
        # Step 1. Initialize variables
        prob_mat = np.zeros((self.n_states, n_obs))
       
        # Step 2. Calculate probabilities
        obs_0 = self.observation_states_dict[input_observation_states[0]]

        for s in range(self.n_states):
            prob_mat[s, 0] = self.prior_p[s] * self.emission_p[s, obs_0]
        
        for t in range(1, n_obs):
            obs_t = self.observation_states_dict[input_observation_states[t]]

            for s in range(self.n_states):
                prob_mat[s, t] = np.sum(prob_mat[:, t-1] * self.transition_p[:, s] * self.emission_p[s, obs_t])
            
        # Step 3. Return final probability 
        forward_probability = np.sum(prob_mat[:, n_obs-1])

        return forward_probability

    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        decode_observation_states = np.atleast_1d(decode_observation_states)

        # Step 1. Initialize variables
        n_obs = decode_observation_states.size
        if n_obs == 0: 
            raise ValueError('Input sequence is empty.')
        for state in np.unique(decode_observation_states):
            if state not in self.observation_states:
                raise ValueError(f'Invalid state: {state}')

        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((n_obs, self.hidden_states.size))

        #store best path for traceback
        backpointer = np.empty((n_obs, self.hidden_states.size), dtype=int)

       
        # Step 2. Calculate Probabilities
        obs_0 = self.observation_states_dict[decode_observation_states[0]]

        for s in range(self.n_states):
            viterbi_table[0, s] = self.prior_p[s] * self.emission_p[s, obs_0]
            backpointer[0, s] = 0

        for t in range(1, n_obs):
            obs_t = self.observation_states_dict[decode_observation_states[t]]
            
            for s in range(self.n_states):
                probs = viterbi_table[t-1, :] * self.transition_p[:, s] * self.emission_p[s, obs_t]

                viterbi_table[t, s] = np.max(probs)
                backpointer[t, s] = np.argmax(probs)
        
        # Step 3. Traceback 
        best_path_pointer = np.argmax(viterbi_table[n_obs-1, :])
        best_hidden_state_sequence = [self.hidden_states_dict[best_path_pointer]]

        for t in range(n_obs-1, 0, -1):
            best_path_pointer = backpointer[t, best_path_pointer]
            best_hidden_state_sequence = [self.hidden_states_dict[best_path_pointer]] + best_hidden_state_sequence

      
        # Step 4. Return best hidden state sequence 
        return best_hidden_state_sequence