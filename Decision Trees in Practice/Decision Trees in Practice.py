

import graphlab


# # Load LendingClub Dataset

# This assignment will use the [LendingClub](https://www.lendingclub.com/) dataset used in the previous two assignments.


loans = graphlab.SFrame('lending-club-data.gl/')


# As before, we reassign the labels to have +1 for a safe loan, and -1 for a risky (bad) loan.


loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')


# We will be using the same 4 categorical features as in the previous assignment: 
# 1. grade of the loan 
# 2. the length of the loan term
# 3. the home ownership status: own, mortgage, rent
# 4. number of years of employment.
# 
# In the dataset, each of these features is a categorical feature. Since we are building a binary decision tree, we will have to convert this to binary data in a subsequent section using 1-hot encoding.


features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'
loans = loans[features + [target]]


# ## Subsample dataset to make sure classes are balanced

# Just as we did in the previous assignment, we will undersample the larger class (safe loans) in order to balance out our dataset. This means we are throwing away many data points. We used `seed = 1` so everyone gets the same results.


safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Since there are less risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed = 1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)



loans_data = risky_loans.append(safe_loans)
for feature in features:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})    
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)
    
    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data.remove_column(feature)
    loans_data.add_columns(loans_data_unpacked)


# The feature columns now look like this:


features = loans_data.column_names()
features.remove('safe_loans')  # Remove the response variable
features


# ## Train-Validation split
# 
# We split the data into a train-validation split with 80% of the data in the training set and 20% of the data in the validation set. We use `seed=1` so that everyone gets the same result.



train_data, validation_set = loans_data.random_split(.8, seed=1)



def reached_minimum_node_size(data, min_node_size):
    # Return True if the number of data points is less than or equal to the minimum node size.
    ## YOUR CODE HERE
    return len(data)<= min_node_size



def error_reduction(error_before_split, error_after_split):
    # Return the error before the split minus the error after the split.
    ## YOUR CODE HERE
    return error_before_split-error_after_split




def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0
    
    # Count the number of 1's (safe loans)
    ## YOUR CODE HERE
    safe = sum(labels_in_node == 1)
    # Count the number of -1's (risky loans)
    ## YOUR CODE HERE
    risky = sum(labels_in_node == -1)
    # Return the number of mistakes that the majority classifier makes.
    ## YOUR CODE HERE
    return min(safe,risky)


# We then wrote a function `best_splitting_feature` that finds the best feature to split on given the data and a list of features to consider.
# 


def best_splitting_feature(data, features, target):
    
    target_values = data[target]
    best_feature = None # Keep track of the best feature 
    best_error = 10     # Keep track of the best error so far 
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    # Loop through each feature to consider splitting on that feature
    for feature in features:
        
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        
        # The right split will have all data points where the feature value is 1
        ## YOUR CODE HERE
        right_split = data[data[feature] == 1]
            
        # Calculate the number of misclassified examples in the left split.
        # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
        # YOUR CODE HERE
        left_mistakes = intermediate_node_num_mistakes(left_split[target])            

        # Calculate the number of misclassified examples in the right split.
        ## YOUR CODE HERE
        right_mistakes = intermediate_node_num_mistakes(right_split[target])
            
        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        ## YOUR CODE HERE
        error = float(left_mistakes+right_mistakes)/len(data)

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        ## YOUR CODE HERE
        if error < best_error:
            best_error = error
            best_feature = feature
    return best_feature # Return the best feature we found


# Finally, recall the function `create_leaf` from the previous assignment, which creates a leaf node given a set of target values.  


def create_leaf(target_values):
    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf':  True   }   ## YOUR CODE HERE
    
    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])
    
    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] =  +1        ## YOUR CODE HERE
    else:
        leaf['prediction'] =   -1       ## YOUR CODE HERE
        
    # Return the leaf node        
    return leaf 



def decision_tree_create(data, features, target, current_depth = 0, 
                         max_depth = 10, min_node_size=1, 
                         min_error_reduction=0.0):
    
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    
    
    # Stopping condition 1: All nodes are of the same type.
    if intermediate_node_num_mistakes(target_values) == 0:
        print "Stopping condition 1 reached. All data points have the same target value."                
        return create_leaf(target_values)
    
    # Stopping condition 2: No more features to split on.
    if remaining_features == []:
        print "Stopping condition 2 reached. No remaining features."                
        return create_leaf(target_values)    
    
    # Early stopping condition 1: Reached max depth limit.
    if current_depth >= max_depth:
        print "Early stopping condition 1 reached. Reached maximum depth."
        return create_leaf(target_values)
    
    # Early stopping condition 2: Reached the minimum node size.
    # If the number of data points is less than or equal to the minimum size, return a leaf.
    if reached_minimum_node_size(data, min_node_size) :      ## YOUR CODE HERE 
        print "Early stopping condition 2 reached. Reached minimum node size."
        return create_leaf(target_values)  ## YOUR CODE HERE
    
    # Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    # Early stopping condition 3: Minimum error reduction
    # Calculate the error before splitting (number of misclassified examples 
    # divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))
    
    # Calculate the error after splitting (number of misclassified examples 
    # in both groups divided by the total number of examples)
    left_mistakes =  intermediate_node_num_mistakes(left_split[target])  ## YOUR CODE HERE
    right_mistakes =  intermediate_node_num_mistakes(right_split[target]) ## YOUR CODE HERE
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))
    
    # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
    if error_reduction(error_before_split, error_after_split) <= min_error_reduction:    ## YOUR CODE HERE
        print "Early stopping condition 3 reached. Minimum error reduction."
        return create_leaf(target_values) ## YOUR CODE HERE 
    
    
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (                      splitting_feature, len(left_split), len(right_split))
    
    
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, 
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)        
    
    ## YOUR CODE HERE
    right_tree = decision_tree_create(right_split, remaining_features, target, 
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)       
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}


# Here is a function to count the nodes in your tree:


def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])


# Run the following test code to check your implementation. Make sure you get **'Test passed'** before proceeding.


small_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth = 2, 
                                        min_node_size = 10, min_error_reduction=0.0)
if count_nodes(small_decision_tree) == 7:
    print 'Test passed!'
else:
    print 'Test failed... try again!'
    print 'Number of nodes found                :', count_nodes(small_decision_tree)
    print 'Number of nodes that should be there : 7' 



my_decision_tree_new = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 100, min_error_reduction=0.0)


# Let's now train a tree model **ignoring early stopping conditions 2 and 3** so that we get the same tree as in the previous assignment.  To ignore these conditions, we set `min_node_size=0` and `min_error_reduction=-1` (a negative value).



my_decision_tree_old = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=-1)




def classify(tree, x, annotate = False):   
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate: 
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction'] 
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
             return classify(tree['right'], x, annotate)
               ### YOUR CODE HERE
                




print 'Predicted class: %s ' % classify(my_decision_tree_new, validation_set[0])


# Let's add some annotations to our prediction to see what the prediction path was that lead to this predicted class:

classify(my_decision_tree_new, validation_set[0], annotate = True)


# Let's now recall the prediction path for the decision tree learned in the previous assignment, which we recreated here as `my_decision_tree_old`.


classify(my_decision_tree_old, validation_set[0], annotate = True)



def evaluate_classification_error(tree, data, target='safe_loans'):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))
    
    # Once you've made the predictions, calculate the classification error and return it
    ## YOUR CODE HERE
    error = sum(prediction != data[target])
    return float(error)/len(data)


# Now, let's use this function to evaluate the classification error of `my_decision_tree_new` on the **validation_set**.


evaluate_classification_error(my_decision_tree_new, validation_set, target)


# Now, evaluate the validation error using `my_decision_tree_old`.



evaluate_classification_error(my_decision_tree_old, validation_set, target)



model_1 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 2, 
                                min_node_size = 0, min_error_reduction=-1)
model_2 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=-1)
model_3 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 14, 
                                min_node_size = 0, min_error_reduction=-1)


# ### Evaluating the models


print "Training data, classification error (model 1):", evaluate_classification_error(model_1, train_data, target)
print "Training data, classification error (model 2):", evaluate_classification_error(model_2, train_data, target)
print "Training data, classification error (model 3):", evaluate_classification_error(model_3, train_data, target)


# Now evaluate the classification error on the validation data.


print "Validation data, classification error (model 1):", evaluate_classification_error(model_1, validation_set)
print "Validation data, classification error (model 2):", evaluate_classification_error(model_2, validation_set)
print "Validation data, classification error (model 3):", evaluate_classification_error(model_3, validation_set)



def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])


# Compute the number of nodes in `model_1`, `model_2`, and `model_3`.

print count_leaves(model_1)
print count_leaves(model_2)
print count_leaves(model_3)




model_4 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=-1)
model_5 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=0)
model_6 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=5)


# Calculate the accuracy of each model (**model_4**, **model_5**, or **model_6**) on the validation set. 


print "Validation data, classification error (model 4):", evaluate_classification_error(model_4, validation_set, target)
print "Validation data, classification error (model 5):", evaluate_classification_error(model_5, validation_set, target)
print "Validation data, classification error (model 6):", evaluate_classification_error(model_6, validation_set, target)




print count_leaves(model_4)
print count_leaves(model_5)
print count_leaves(model_6)




model_7 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=-1)
model_8 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 2000, min_error_reduction=-1)
model_9 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 50000, min_error_reduction=-1)




print "Validation data, classification error (model 4):", evaluate_classification_error(model_7, validation_set, target)
print "Validation data, classification error (model 5):", evaluate_classification_error(model_8, validation_set, target)
print "Validation data, classification error (model 6):", evaluate_classification_error(model_9, validation_set, target)




print count_leaves(model_7)
print count_leaves(model_8)
print count_leaves(model_9)
