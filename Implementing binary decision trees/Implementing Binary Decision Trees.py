

import graphlab


# # Load the lending club dataset

# We will be using the same [LendingClub](https://www.lendingclub.com/) dataset as in the previous assignment.

#

loans = graphlab.SFrame('lending-club-data.gl/')


# Like the previous assignment, we reassign the labels to have +1 for a safe loan, and -1 for a risky (bad) loan.



loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')


# Since we are building a binary decision tree, we will have to convert these categorical features to a binary representation in a subsequent section using 1-hot encoding.



features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'
loans = loans[features + [target]]



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




features = loans_data.column_names()
features.remove('safe_loans')  # Remove the response variable
features



print "Number of features (after binarizing categorical variables) = %s" % len(features)


# Let's explore what one of these columns looks like:



print "Total number of grade.A loans : %s" % loans_data['grade.A'].sum()
print "Expexted answer               : 6422"


# ## Train-test split
# 
# We split the data into a train test split with 80% of the data in the training set and 20% of the data in the test set. We use `seed=1` so that everyone gets the same result.


train_data, test_data = loans_data.random_split(.8, seed=1)


# Now, let us write the function `intermediate_node_num_mistakes` which computes the number of misclassified examples of an intermediate node given the set of labels (y values) of the data points contained in the node. Fill in the places where you find `## YOUR CODE HERE`. There are **three** places in this function for you to fill in.



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
    return min(safe, risky)




def best_splitting_feature(data, features, target):
    
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




if best_splitting_feature(train_data, features, 'safe_loans') == 'term. 36 months':
    print 'Test passed!'
else:
    print 'Test failed... try again!'




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
        leaf['prediction'] =   -1      ## YOUR CODE HERE
        
    # Return the leaf node        
    return leaf 



def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
    if  intermediate_node_num_mistakes(target_values)== 0:  ## YOUR CODE HERE
        print "Stopping condition 1 reached."     
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features == []:   ## YOUR CODE HERE
        print "Stopping condition 2 reached."    
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth >=max_depth :  ## YOUR CODE HERE
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    ## YOUR CODE HERE
    splitting_feature = best_splitting_feature(data, features, target)
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]      ## YOUR CODE HERE
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (                      splitting_feature, len(left_split), len(right_split))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print "Creating leaf node."
        ## YOUR CODE HERE
        return create_leaf(right_split[target])
        
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)        
    ## YOUR CODE HERE
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth)      

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}


# Here is a recursive function to count the nodes in your tree:



def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])


# Run the following test code to check your implementation. Make sure you get **'Test passed'** before proceeding.



small_data_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth = 3)
if count_nodes(small_data_decision_tree) == 13:
    print 'Test passed!'
else:
    print 'Test failed... try again!'
    print 'Number of nodes found                :', count_nodes(small_data_decision_tree)
    print 'Number of nodes that should be there : 13' 



# Make sure to cap the depth at 6 by using max_depth = 6
my_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth=6)


# ## Making predictions with a decision tree
# 
# As discussed in the lecture, we can make predictions from the decision tree with a simple recursive function. Below, we call this function `classify`, which takes in a learned `tree` and a test point `x` to classify.  We include an option `annotate` that describes the prediction path when set to `True`.
# 
# Fill in the places where you find `## YOUR CODE HERE`. There is **one** place in this function for you to fill in.


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


# Now, let's consider the first example of the test set and see what `my_decision_tree` model predicts for this data point.




print 'Predicted class: %s ' % classify(my_decision_tree, test_data[0])


# Let's add some annotations to our prediction to see what the prediction path was that lead to this predicted class:


classify(my_decision_tree, test_data[0], annotate=True)


# This function should calculate a prediction (class label) for each row in `data` using the decision `tree` and return the classification error computed using the above formula. Fill in the places where you find `## YOUR CODE HERE`. There is **one** place in this function for you to fill in.


def evaluate_classification_error(tree, data, target):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))
    
    # Once you've made the predictions, calculate the classification error and return it
    ## YOUR CODE HERE
    error = sum(prediction != data[target])
    return float(error)/len(data)


# Now, let's use this function to evaluate the classification error on the test set.


evaluate_classification_error(my_decision_tree, test_data, target)


# **Quiz Question:** Rounded to 2nd decimal point, what is the classification error of **my_decision_tree** on the **test_data**?

# ## Printing out a decision stump

# As discussed in the lecture, we can print out a single decision stump (printing out the entire tree is left as an exercise to the curious reader). 


def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('.')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)'         % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))



print_stump(my_decision_tree)



print_stump(my_decision_tree['left'], my_decision_tree['splitting_feature'])


# ### Exploring the left subtree of the left subtree
# 


print_stump(my_decision_tree['left']['left'], my_decision_tree['left']['splitting_feature'])



print_stump(my_decision_tree['right'], my_decision_tree['splitting_feature'])



print_stump(my_decision_tree['right']['right'], my_decision_tree['left']['splitting_feature'])


