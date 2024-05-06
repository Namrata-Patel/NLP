import gpt_2_simple as gpt2

# Download and load the GPT-2 model (this downloads the model if not already downloaded)
gpt2.download_gpt2(model_name="124M")

# Start a TensorFlow session
sess = gpt2.start_tf_sess()

# Provide some context
context = "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity."

# Generate answers to questions using GPT-2
questions = [
    "Who developed the theory of relativity?",
    "What is Albert Einstein known for?"
]

for question in questions:
    answer = gpt2.generate(sess, prefix=context + "\nQuestion: " + question, return_as_list=True)[0]
    print("Question:", question)
    print("Answer:", answer.strip())
    print()
