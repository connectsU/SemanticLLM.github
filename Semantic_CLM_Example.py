import openai

# Causal Language Modeling (CLM) is a foundational technique in natural language processing (NLP) used to train models to predict the next word in a sequence based on the preceding context. 
# This approach is central to autoregressive models like GPT (Generative Pre-trained Transformer), which generate coherent and contextually relevant text by learning from vast corpora of language data.

# Set your OpenAI API key
openai.api_key = "your-api-key"

def predict_next_words(prompt, max_tokens=10, temperature=0.7, model="text-davinci-003"):
    """
    Predicts the next words in a sequence using OpenAI's GPT model.
    
    Parameters:
        prompt (str): The input text sequence.
        max_tokens (int): Number of tokens to generate.
        temperature (float): Controls randomness (0.0 = deterministic, 1.0 = creative).
        model (str): OpenAI model to use (e.g., 'text-davinci-003').
    
    Returns:
        str: The generated continuation.
    """
    try:
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            n=1,
            stop=None
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {e}"

# Example usage
if __name__ == "__main__":
    input_text = "The future of artificial intelligence is"
    continuation = predict_next_words(input_text, max_tokens=15)
    print(f"Input: {input_text}")
    print(f"Generated: {continuation}")
