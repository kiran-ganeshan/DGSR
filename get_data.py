import pandas as pd
data = pd.read_csv('/data/groups/dgnn/raw_data/enrollments.csv')
data.reset_index(drop=True, inplace=True)
data.drop(columns='grade')
data.rename(columns={'semester_index': 'time', 'ppsk': 'user_id', 'cid': 'item_id'}, inplace=True)
data = data[['user_id', 'item_id', 'time']]
data.to_csv('./Data/Enrollments.csv', index=False)