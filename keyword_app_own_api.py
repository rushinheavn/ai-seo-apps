import streamlit as st
import asyncio
import aiohttp
import ssl
import certifi
import nest_asyncio
import time
import pandas as pd

# Allow nested event loops in Streamlit
nest_asyncio.apply()

# Set default values
DEFAULT_BATCH_SIZE = 10

# Default prompt template (uses double curly braces for replacement)
default_prompt_template = """based on the description for AIRMDR

AirMDR, a cybersecurity company specializing in AI-powered Managed Detection and Response (MDR) services.

AirMDR provides AI-driven MDR solutions that help businesses detect, investigate, and respond to cybersecurity threats quickly. They do not sell standalone cybersecurity tools but instead offer a fully managed, AI-enhanced security monitoring service. Their solutions integrate with EDR, XDR, NDR, SIEM, and other security tools to automate threat detection and response.
You need to categorize this keyword - {{cell_value}} into one of the following categories.

SOAR  
MDR General  
MDR Solutions  
AI  
SOC  
Competitor Brand  
AirMDR Brand  
SIEM  
Automation

Just output one of the above categories as output – nothing else – select the most relevant keyword as per the {{cell_value}} keyword"""


async def process_text(session, text,
                       prompt_template, api_key):
    """
    Replace the placeholder in the prompt template with the given text,
    then call the GPT API asynchronously.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # Replace the placeholder; note we use the double curly braces as in the default template.
    prompt = prompt_template.replace(
        "{{cell_value}}", text)
    payload = {
        "model": "gpt-4o",
        # Change model if needed
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant trained to categorize keywords"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7
    }

    try:
        async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload) as response:
            response.raise_for_status()
            response_json = await response.json()
            # Extract the GPT response text
            return text, \
            response_json['choices'][0][
                'message']['content']
    except Exception as e:
        return text, f"Error: {str(e)}"


async def run_gpt(keywords, prompt_template,
                  batch_size, api_key):
    """
    Process the list of keywords in batches and return a list of tuples.
    Each tuple contains the input keyword and the GPT output.
    """
    results = []
    ssl_context = ssl.create_default_context(
        cafile=certifi.where())
    async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                    ssl=ssl_context)) as session:
        # Process keywords in batches
        for i in range(0, len(keywords),
                       batch_size):
            batch = keywords[i:i + batch_size]
            tasks = [process_text(session, text,
                                  prompt_template,
                                  api_key) for
                     text in batch]
            batch_results = await asyncio.gather(
                *tasks)
            results.extend(batch_results)
    return results


def main():
    st.title("GPT Keyword Categorizer")
    st.write(
        "Customize the prompt template, enter keywords (one per line), and click **Run** to see the output.")

    # API key input (using password input for security)
    api_key = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="Your API key will not be stored")

    # Sidebar or main area inputs
    prompt_template = st.text_area(
        "Prompt Template",
        value=default_prompt_template, height=300)
    keywords_input = st.text_area(
        "Input Keywords (one per line)",
        value="keyword1\nkeyword2", height=150)
    batch_size = st.number_input("Batch Size",
                                 min_value=1,
                                 max_value=100,
                                 value=DEFAULT_BATCH_SIZE,
                                 step=1)

    if st.button("Run"):
        # Check if API key is provided
        if not api_key:
            st.error(
                "Please enter your OpenAI API Key.")
            return

        # Split keywords by line and remove any blank lines
        keywords = [line.strip() for line in
                    keywords_input.splitlines() if
                    line.strip()]
        if not keywords:
            st.error(
                "Please enter at least one keyword.")
            return

        start_time = time.time()
        with st.spinner("Processing keywords..."):
            # Run the async GPT calls with the provided API key
            results = asyncio.run(
                run_gpt(keywords, prompt_template,
                        batch_size, api_key))
        end_time = time.time()
        st.success(
            f"Processing completed in {end_time - start_time:.2f} seconds.")

        # Create a DataFrame from the results and display it
        df = pd.DataFrame(results,
                          columns=["Input",
                                   "Output"])
        st.dataframe(df)

        # Offer a CSV download of the results
        csv = df.to_csv(index=False).encode(
            'utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='output.csv',
            mime='text/csv'
        )


if __name__ == "__main__":
    main()