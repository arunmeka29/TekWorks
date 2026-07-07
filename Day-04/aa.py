import streamlit as st
#title
st.title("Hello, Streamlit!")

#subheader
st.subheader("Welcome to your first Streamlit app.")

#horizontal line
st.markdown("--------------------------------------------------------------")

#text
st.write("This application allows you to perform CRUD operations.")

#write method
st.write("Hello Arun")
st.write([1,2,3])

#caption
st.caption("This is a caption providing additional context.")

#code
st.code(""" 
        def add(a,b):
            return a + b
        """, language='python')
#latex
st.latex(r''' e^{i\pi} + 1 = 0 ''')

st.divider()    

if st.button("Click Me"):
    st.write("Button Clicked!")
    st.success("Success! You clicked the button.")
    st.balloons()
if st.button("Celebrate"):
    st.snow()
else:
    st.write("Button not clicked yet.")
    st.error("Error! You haven't clicked the button yet.")
    
#text input method to get user input
try:
    age = st.text_input("Enter your age:")
    age = int(age)
    st.write(f"You are {age} years old.")
    if age < 0:
        st.warning("Age cannot be negative.")
    elif age<18:
        st.info("you are minor.")
    else:
        st.success("You are an adult.")
except ValueError:
    st.error("Please enter a valid age.")

feedback=st.text_area("Provide your feedback here:")
st.write(feedback)

if st.checkbox("I agree to the terms and conditions"):
    st.write("Thank you for agreeing!") 

gender = st.radio("Select your gender:", ("Male", "Female", "Other"))
st.write(f"You selected: {gender}") 


