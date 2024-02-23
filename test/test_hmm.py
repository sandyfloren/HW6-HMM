import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    HMM = HiddenMarkovModel(mini_hmm['observation_states'], mini_hmm['hidden_states'], mini_hmm['prior_p'], mini_hmm['transition_p'], mini_hmm['emission_p'])   
   

    # Forward algorithm

    assert HMM.forward(mini_hmm['observation_states'][0]) == HMM.prior_p[0] * HMM.emission_p[0, 0] + HMM.prior_p[1] * HMM.emission_p[1, 0]

    assert HMM.forward(mini_hmm['observation_states'][1]) == HMM.prior_p[0] * HMM.emission_p[0, 1] + HMM.prior_p[1] * HMM.emission_p[1, 1]

    # Check for length 0 input sequence edge case 
    with pytest.raises(ValueError):
        HMM.forward(np.array([]))

    # Check for input sequence containing invalid state edge case  
    with pytest.raises(ValueError):
        HMM.forward(np.array(['blustery']))


    # Viterbi algorithm

    viterbi = HMM.viterbi(mini_input['observation_state_sequence'])

    # Check that viterbi output is correct
    assert viterbi == [x for x in mini_input['best_hidden_state_sequence']]

    assert HMM.viterbi(mini_hmm['observation_states'][0]) == np.array(mini_hmm['hidden_states'][np.argmax(mini_hmm['emission_p'][:, 0])])

    # Check for length 0 input sequence edge case 
    with pytest.raises(ValueError):
        HMM.viterbi(np.array([]))

    # Check for input sequence containing invalid state edge case  
    with pytest.raises(ValueError):
        HMM.viterbi(np.array(['blustery']))




def test_full_weather():


    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')


    HMM = HiddenMarkovModel(full_hmm['observation_states'], full_hmm['hidden_states'], full_hmm['prior_p'], full_hmm['transition_p'], full_hmm['emission_p'])

    viterbi = HMM.viterbi(full_input['observation_state_sequence'])
    
    # Check that viterbi output is correct
    assert viterbi == [x for x in full_input['best_hidden_state_sequence']]







