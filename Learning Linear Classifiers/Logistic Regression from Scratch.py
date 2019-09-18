


from __future__ import division
import graphlab
import math
import string
import json
import numpy as np
from math import sqrt

products = graphlab.SFrame('amazon_baby_subset.gl/')


with open('important_words.json', 'r') as f: 
    important_words = json.load(f)
important_words = [str(s) for s in important_words]


def remove_punctuation(text):
    return text.translate(None, string.punctuation) 

products['review_clean'] = products['review'].apply(remove_punctuation)



for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))

def get_numpy_data(data_sframe, features, label):
    data_sframe['intercept'] = 1
    features = ['intercept'] + features
    features_sframe = data_sframe[features]
    feature_matrix = features_sframe.to_numpy()
    label_sarray = data_sframe[label]
    label_array = label_sarray.to_numpy()
    return(feature_matrix, label_array)







def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients  
    # YOUR CODE HERE
    
    product=np.dot(feature_matrix, coefficients)
    
    # Compute P(y_i = +1 | x_i, w) using the link function
    # YOUR CODE HERE
    predictions=1.0/(1. + np.exp(-product))
    # return predictions
    return predictions


def feature_derivative(errors, feature):     
    # Compute the dot product of errors and feature
    derivative = np.dot(errors,feature)
    
    # Return the derivative
    return derivative



def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    
    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    
    lp = np.sum((indicator-1)*scores - logexp)
    return lp



def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in xrange(max_iter):

        # Predict P(y_i = +1|x_i,w) using your predict_probability() function
        # YOUR CODE HERE
        predictions = predict_probability(feature_matrix, coefficients)
        
        # Compute indicator value for (y_i = +1)
        indicator = (sentiment==+1)
        
        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)): # loop over each coefficient
            
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j].
            # Compute the derivative for coefficients[j]. Save it in a variable called derivative
            # YOUR CODE HERE
            derivative = feature_derivative(errors, feature_matrix[:,j] )
            
            # add the step size times the derivative to the current coefficient
            ## YOUR CODE HERE
            coefficients[j]=coefficients[j]+(derivative*step_size)
        
        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0)         or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' %                 (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients


# Now, let us run the logistic regression solver.


coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients=np.zeros(194),
                                   step_size=1e-7, max_iter=301)
print coefficients


scores = np.dot(feature_matrix, coefficients)
print len(scores)
print sum(scores>0)
print sum(scores<=0)



a=+1 if scores.all>0 else -1
for i in range(len(scores)):
    scores[i]=+1 if scores[i]>0 else -1
scores


correctly_classify = sum(scores==sentiment)


num_mistakes = len(products) - correctly_classify
accuracy = correctly_classify/len(products)
print "-----------------------------------------------------"
print '# Reviews   correctly classified =', correctly_classify
print '# Reviews incorrectly classified =', num_mistakes
print '# Reviews total                  =', len(products)
print "-----------------------------------------------------"
print 'Accuracy = %.2f' % accuracy

