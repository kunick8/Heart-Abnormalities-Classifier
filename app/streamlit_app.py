import streamlit as st

home = st.Page('home_page.py', title='Home Page', icon ='🏠')
overview = st.Page("overview.py", title="Overview", icon="❤️")
performance = st.Page("model_performance.py", title="Model Performance", icon="📊")
predict_page = st.Page('predict_page.py', title = 'Predict', icon = '🎯')

pg = st.navigation([home, overview, performance, predict_page])


pg.run()
