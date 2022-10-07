##### IMPORTS #####
###This uses tweepy-
import tweepy

# Read Twitter API key from the private key file
with open('./TwitterApiKey.txt', 'r') as TwitterApiKeyFile:
    APIkey = TwitterApiKeyFile.readline()


def RecentTweetsWrapper(Query: str):
    bearer_token = str(APIkey)

    client = tweepy.Client(bearer_token)

    # Search Recent Tweets

    # This endpoint/method returns Tweets from the last seven days
    response = client.search_recent_tweets((f'"{str(Query)}" -is:retweet lang:en'), max_results=10, tweet_fields=['author_id'])
    # The method returns a Response object, a named tuple with data, includes,
    # errors, and meta fields
    # print(response.meta)

    # In this case, the data field of the Response returned is a list of Tweet
    # objects
    tweets = response.data
    # Each Tweet object has default ID and text fields
    formatted_tweets = []

    if tweets:
        for tweet in tweets:
            author_username = ResolveAuthorUsernameFromID(tweet.author_id)
            tweet_data = (tweet.id, tweet.text, author_username)
            formatted_tweets.append(tweet_data)

        # print(f'{tweet.id} {tweet.text} {tweet.author_id}')

    return formatted_tweets
    # By default, this endpoint/method returns 10 results
    # You can retrieve up to 100 Tweets by specifying max_results
    # response = client.search_recent_tweets("Covid", max_results=100)


def ResolveAuthorUsernameFromID(ID):
    bearer_token = str(APIkey)

    client = tweepy.Client(bearer_token)

    # By default, only the ID, name, and username fields of each user will be
    # returned
    # Additional fields can be retrieved using the user_fields parameter 
    response = client.get_users(ids=ID)

    for user in response.data:
        return user.username


if __name__ == "__main__":
    ReturnedTweets = RecentTweetsWrapper("covid and is")
    for i in ReturnedTweets:
        print(i)
    print('###########################')
