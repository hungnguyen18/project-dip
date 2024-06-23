import streamlit as st
from st_pages import Page, show_pages, add_page_title

# Optional -- adds the title and icon to the current page
add_page_title()

# Specify what pages should be shown in the sidebar, and what their titles
# and icons should be
show_pages(
    [
        Page("app.py", "Home", "🏠"),
        Page("pages/face-detection.py", "Nhận diện khuôn mặt", "🤦"),
        Page("pages/fruit-identification.py", "Nhận diện trái cây", "🍎"),
        Page("pages/recognize-handwritten-digits.py", "Nhận dạng chữ số MNIST", "🖍️"),
        Page("pages/sample.py", "sample", "🖍️"),
    ]
)


st.title("Project digital image processing")

st.write("# Welcome to Streamlit! 👋")


st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)
