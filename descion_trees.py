import pandas as pd

data = {
    'Age': [25, 45, 35, 50, 40, 60, 30],
    'Income': [50000, 100000, 75000, 80000, 90000, 150000, 60000],
    'Bought Product?': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

def ginni_impurity(data):
  
  class_count = data["Bought Product?"].value_counts()
  total = len(data)

  gini = 1 - sum((count/total) ** 2 for count in class_count)

  return gini

def split_data(data,feature,threshold):
  
  left_data = data[data[feature] <= threshold]
  right_data = data[data[feature] > threshold]

  return left_data,right_data


def best_split(data,feature):
  best_gini = float('inf')
  best_threshold, best_left,best_right = None, None,None

  for threshold in data[feature].unique():
    left_data,right_data = split_data(data,feature,threshold)

    gini_left = ginni_impurity(left_data)
    gini_right = ginni_impurity(right_data)


    gini = (len(left_data)/ len(data)) * gini_left + (len(right_data)/ len(data)) * gini_right

    if gini < best_gini:
      best_gini = gini
      best_threshold = threshold
      best_left = left_data
      best_right = right_data


    return best_gini,best_threshold,best_left,best_right



def build_tree(data,depth=0):
  
  best_gini = float('inf')
  best_feature = None
  best_threshold = None
  best_left = None
  best_right = None

  for feature in data.columns[:-1]:
    gini,threshold,left_data,right_data = best_split(data,feature)


    if gini < best_gini:
      best_gini = gini
      best_feature = feature
      best_threshold = threshold
      best_left = left_data
      best_right = right_data

  if best_gini == float('inf'):
    return None


  left_tree = build_tree(best_left,depth+1)
  right_tree = build_tree(best_right,depth+1)

  return {
      'feature':best_feature,
      'threshold':best_threshold,
      'left':left_tree,
      'right':right_tree
  }



def predict(tree, row):
    if tree is None:
        return None
    
    if isinstance(tree, dict) and tree.get('left') is None and tree.get('right') is None:
        return row['Bought Product?'].mode()[0]
    
    if row[tree['feature']] <= tree['threshold']:
        return predict(tree['left'], row)
    else:
        return predict(tree['right'], row)


tree = build_tree(df, max_depth=3)

print("Decision Tree:", tree)

sample = pd.DataFrame({
    'Age': [30],
    'Income': [70000]
})

prediction = predict(tree, sample.iloc[0])
print(f"Prediction for {sample.iloc[0].to_dict()}: {prediction}")
