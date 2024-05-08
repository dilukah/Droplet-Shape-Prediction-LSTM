import numpy as np

def check_matched_sequences(prediction, ground_truth):
    
    prediction_boolean = prediction>0.5
    prediction_binary = np.where(prediction_boolean, 1, 0)
    prediction_binary = prediction_binary
    unmatched_SequenceSteps = np.any(prediction_binary!=ground_truth,axis=2)
    unmatched_Sequence =  np.any( unmatched_SequenceSteps == True,axis=1)
    unmatched_Sequence_idx = np.where(np.any( unmatched_SequenceSteps == True,axis=1))
    #print("Unmatched sequence index: ", unmatched_Sequence_idx)
    matchedSequences = ~unmatched_Sequence
    matchPercentage = np.sum(matchedSequences)/len(matchedSequences)*100
    print(" Matched Sequences: ", np.sum(matchedSequences),"/",len(matchedSequences), " ",  matchPercentage)
    return unmatched_Sequence_idx, matchedSequences, matchPercentage

def check_matched_actuations(prediction, ground_truth):
    prediction_boolean = prediction>0.5
    prediction_binary = np.where(prediction_boolean, 1, 0)
    unmatched_actuation_combinations = np.any(prediction_binary!=ground_truth,axis=1)
    unmatched_actuation_idx = np.where(unmatched_actuation_combinations == True)
    print("Unmatched actuation index: ", unmatched_actuation_idx)
    matched_actuations = ~unmatched_actuation_combinations
    matchPercentage = np.sum(matched_actuations)/len(matched_actuations)*100

    print(" Matched Sequences: ", np.sum(matched_actuations),"/",len(matched_actuations), " ",  np.sum(matched_actuations)/len(matched_actuations)*100)

def load_all_data_files_and_stack_data(data_files):
    x_data = [np.load(file)['x'] for file in data_files]
    y_data = [np.load(file)['y'] for file in data_files]
    x_data = np.concatenate( x_data, axis=0 )
    y_data = np.concatenate( y_data, axis=0 )
    print('Shape of x_data', x_data.shape)
    print('Shape of y_data', y_data.shape)
    return x_data, y_data

def train_test_data_split(x_data, y_data, number_of_validation_tests, preffered_test_index = None):
    n_samples = x_data.shape[0]
    test_data_idx = np.random.permutation(np.arange(0,n_samples))[:number_of_validation_tests]
    x_test, y_test = x_data[test_data_idx,:], y_data[test_data_idx,:]
    #Delete the validation set from test data
    x_train = np.delete(x_data, test_data_idx, 0)
    y_train = np.delete(y_data, test_data_idx, 0)
    #print('Random index: ', test_data_idx)
    return x_train, y_train, x_test, y_test, test_data_idx