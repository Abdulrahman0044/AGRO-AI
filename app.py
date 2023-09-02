import streamlit as st
import openai

openai.api_key = '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'
def ask_openai(question):
    response = openai.Completion.create(
        engine="davinci",
        prompt=question, 
        max_tokens=100,
    )
    return response.choices[0].text.strip()

def main():
    st.title("Agriculture Assistant for Africa")

    user_input = st.text_input("Ask a question about agriculture in Africa:")
    if st.button("Get Answer"):
        answer = ask_openai(user_input)
        st.write(answer)
if __name__ == "__main__":
    main()