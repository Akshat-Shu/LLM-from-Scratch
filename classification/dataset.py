import pandas as pd

def get_balanced_dataset(df):
    num_spam = df['label'].value_counts()['spam']
    not_spam_subset = df[df['label'] == 'ham'].sample(num_spam, random_state=123)
    balanced_df = pd.concat([df[df['label'] == 'spam'], not_spam_subset])
    return balanced_df

def random_split(df, train_frac, val_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * val_frac)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df
