import boto3
import json
import time

# Netflix AI Recommender Engine with High Availability
class NetflixAI:
    def __init__(self):
        self.client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
        self.primary_model = "anthropic.claude-3-sonnet-20240229-v1:0"
        self.fallback_model = "anthropic.claude-3-haiku-20240307-v1:0"

    def get_movie_recommendation(self, user_mood, retry_count=0):
        prompt = f"User is feeling: {user_mood}. Suggest 3 Netflix movies with reasons."
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}]
        })

        try:
            # Using Primary Model
            model = self.primary_model if retry_count == 0 else self.fallback_model
            print(f"Calling Model: {model}")
            
            response = self.client.invoke_model(body=body, modelId=model)
            response_body = json.loads(response.get('body').read())
            return response_body['content'][0]['text']

        except Exception as e:
            if retry_count < 1:
                print("Primary model failed. Switching to Fallback...")
                return self.get_movie_recommendation(user_mood, retry_count + 1)
            else:
                return "Service temporarily unavailable. Please try again later."

# Example Usage
if __name__ == "__main__":
    recommender = NetflixAI()
    print(recommender.get_movie_recommendation("I want something dark and mysterious like Sacred Games."))
