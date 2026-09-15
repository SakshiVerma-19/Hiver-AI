import os
import pandas as pd
def reconstruct_threads(input_path:str, output_path: str, target_brand:str="AmazonHelp", sample_size:int=5000):
    """
    Parses raw TWCS dataset, extracts target brand interactions, 
    reconstructs turn-by-turn conversations, and saves a cleaned dataset.
    """
    print(f"Loading raw Dataset from {input_path}...")
    cols_to_use = ['tweet_id', 'author_id', 'inbound', 'created_at', 'text', 'in_response_to_tweet_id']
    # Load required columns to save RAM
    df=pd.read_csv(input_path, usecols=cols_to_use)
    print(f"Total raw tweets loaded: {len(df)}")

    df['tweet_id']=df['tweet_id'].astype(str)
    df['in_response_to_tweet_id']=df['in_response_to_tweet_id'].fillna('').astype(str).str.replace(r'\.0$', '' ,regex=True)
    tweet_dict=df.set_index('tweet_id').to_dict('index')
    print(f"Filtering dialouge threads for brand:@{target_brand}")
    reconstructed_threads=[]
    brand_tweets=df[(df['author_id']==target_brand)&(df['inbound']==False)]

    for _, brand_row in brand_tweets.iterrows():
        parent_id=brand_row['in_response_to_tweet_id']
        if parent_id in tweet_dict:
            customer_tweet=tweet_dict[parent_id]
            if customer_tweet['inbound']:
                clean_customer_text=customer_tweet['text'].replace('\n', ' ').strip()
                clean_brand_text=brand_row['text'].replace('\n', ' ').strip()
                reconstructed_threads.append({
                    'customer_tweet_id': parent_id,
                    'brand_tweet_id': brand_row['tweet_id'],
                    'customer_author': customer_tweet['author_id'],
                    'customer_text': clean_customer_text,
                    'brand_response': clean_brand_text,
                    'created_at': brand_row['created_at']
                })
                
        # Limit processed samples for fast local development
        if len(reconstructed_threads) >= sample_size:
            break

    result_df = pd.DataFrame(reconstructed_threads)
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    
    print(f"Successfully processed {len(result_df)} paired conversation threads.")
    print(f"Cleaned output saved to: {output_path}")

if __name__ == "__main__":
    RAW_DATA_PATH = "data/twcs.csv"
    OUTPUT_DATA_PATH = "data/raw_sample.csv"
    
    reconstruct_threads(
        input_path=RAW_DATA_PATH,
        output_path=OUTPUT_DATA_PATH,
        target_brand="AmazonHelp",
        sample_size=5000
    )


