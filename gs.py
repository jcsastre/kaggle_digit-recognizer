# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Function to create model, required for KerasClassifier
def create_model(optimizer='adam'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

if __name__ == "__main__":
    
    # load dataset
    dataset = numpy.loadtxt("./input/pima-indians-diabetes.csv", delimiter=",")

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]

    # create model
    model = KerasClassifier(
        build_fn = create_model, 
        verbose = 0
    )

    grid = GridSearchCV(
        estimator = model, 
        param_grid = dict(
            epochs = [5, 10, 15],
            batch_size = [5, 10, 20],
            optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        ), 
        n_jobs=1, 
        verbose=10
    )

    grid_result = grid.fit(X, Y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))