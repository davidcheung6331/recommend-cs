import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import streamlit as st
import os
from PIL import Image

st.set_page_config(
    page_title="Recommendations by embedding",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

image = Image.open("shopping-banner.png")
st.image(image, caption='created by MJ')


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

system_openai_api_key = os.environ.get('OPENAI_API_KEY')
system_openai_api_key = st.text_input(":key: OpenAI Key :", value=system_openai_api_key)
os.environ["OPENAI_API_KEY"] = system_openai_api_key

openai.api_key = system_openai_api_key

customer_input = "Hi! Can you recommend a good moisturizer for me?"
st.subheader(f":green[ 👩🏽‍🦰" +  customer_input +"]")

if st.button('Yes, we start recommend action'):
    ##################    
    # PRODUCT
    ##################

    # defines a list of dictionaries representing product data
    product_data = [
    {
        "prod_id": 1,
        "prod": "moisturizer",
        "brand":"Aveeno",
        "description": "for dry skin"
    },
    {
        "prod_id": 2,
        "prod": "foundation",
        "brand":"Maybelline",
        "description": "medium coverage"
    },
    {
        "prod_id": 3,
        "prod": "moisturizer",
        "brand":"CeraVe",
        "description": "for dry skin"
    },
    {
        "prod_id": 4,
        "prod": "nail polish",
        "brand":"OPI",
        "description": "raspberry red"
    },
    {
        "prod_id": 5,
        "prod": "concealer",
        "brand":"Chanel",
        "description": "medium coverage"
    },
    {
        "prod_id": 6,
        "prod": "moisturizer",
        "brand":"Ole Henkrisen",
        "description": "for oily skin"
    },
    {
        "prod_id": 7,
        "prod": "moisturizer",
        "brand":"CeraVe",
        "description": "for normal to dry skin"
    },
    {
        "prod_id": 8,
        "prod": "moisturizer",
        "brand":"First Aid Beauty",
        "description": "for dry skin"
    },{
        "prod_id": 9,
        "prod": "makeup sponge",
        "brand":"Sephora",
        "description": "super-soft, exclusive, latex-free foam"
    }]




    st.subheader(":blue[Step 1. 🎉 PRODUCT EMBEDDING]")
    product_data_df = pd.DataFrame(product_data)
    st.write('1.0 create product dataframe')
    # st.write(product_data_df)

    product_data_df['combined'] = product_data_df.apply(lambda row: f"{row['brand']}, {row['prod']}, {row['description']}", axis=1)
    st.write('1.1 create combined column on product data ')
    # st.write(product_data_df)

    with st.spinner('Creating Product Embedding'):
        product_data_df['text_embedding'] = product_data_df.combined.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
        st.write('1.2 product_data with embedding')
        st.write(product_data_df)


    ##################    
    # CUSTOMER ORDER
    ##################
    st.subheader(":blue[Step 2.  💳 CUSTOMER ORDER HISTORY EMBEDDING]")
    customer_order_data = [
    {
        "prod_id": 1,
        "prod": "moisturizer",
        "brand":"Aveeno",
        "description": "for dry skin"
    },{
        "prod_id": 2,
        "prod": "foundation",
        "brand":"Maybelline",
        "description": "medium coverage"
    },{
        "prod_id": 4,
        "prod": "nail polish",
        "brand":"OPI",
        "description": "raspberry red"
    },{
        "prod_id": 5,
        "prod": "concealer",
        "brand":"Chanel",
        "description": "medium coverage"
    },{
        "prod_id": 9,
        "prod": "makeup sponge",
        "brand":"Sephora",
        "description": "super-soft, exclusive, latex-free foam"
    }]

    customer_order_df = pd.DataFrame(customer_order_data)      
    st.write('2.0 create customer_order dataframe ')
    # st.write(customer_order_df)  

    customer_order_df['combined'] = customer_order_df.apply(lambda row: f"{row['brand']}, {row['prod']}, {row['description']}", axis=1)
    st.write('2.1 create combined column on customer order data') 
    st.write(customer_order_df)  

    with st.spinner('Creating Customer Orders History Embedding'):
        customer_order_df['text_embedding'] = customer_order_df.combined.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
        st.write('2.1 customer_order embeding')   
        # st.write(customer_order_df)  


    #############################################    
    # CUSTOMER INQUIRY VS CUSTOMER ORDER HISTORY
    ############################################### 
    st.subheader(":blue[🔍Step 3. CUSTOMER INQUIRY VS CUSTOMER ORDER HISTORY]")
    
    st.write('3.0 customer inquery embedding')
    st.write(customer_input)
    
    response = openai.Embedding.create(
        input=customer_input,
        model="text-embedding-ada-002"
    )
    embeddings_customer_question = response['data'][0]['embedding']
    st.write('3.1 customer inquery embedding')
    # st.write(embeddings_customer_question)


    # compare the user chat input embeddings with the previous product purchases database embeddings

    # Create a new column in the previous purchase product data DataFrame for the search score 
    #  call cosine_similarity for each embedding
    with st.spinner('Creating Customer inquiry + Customer Orders history Embedding'):
        customer_order_df['search_purchase_history'] = customer_order_df.text_embedding.apply(lambda x: cosine_similarity(x, embeddings_customer_question))
        st.write('4.0 customer inquiry VS customer orders embedding with desc order')

        # Sort the DataFrame in descending order based on the highest score:
        st.write('4.1 Sort the DataFrame in descending order based on the highest score:')
        st.write(customer_order_df)

    #############################################    
    # SEARCH CUSTOMER INQUIRY VS PRODUCT 
    ############################################### 
    st.subheader(":blue[Step 4. 🔍 CUSTOMER INQUIRY VS PRODUCT]")
    product_data_df['search_products'] = product_data_df.text_embedding.apply(lambda x: cosine_similarity(x, embeddings_customer_question))
    product_data_df = product_data_df.sort_values('search_products', ascending=False)
    st.write('5.0 customer inquiry VS product embedding')
    st.write(product_data_df)


    #############################################    
    # 2 Dataframes for previously bought products with the highest similarity scores:
    ############################################### 
    st.subheader(":blue[Step 5. 🔍 Top 3 matched Customer Orders and Products]")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(':red[🎉 Top 3 matched customer order]')
        top_3_purchases_df = customer_order_df.head(3)
        st.write(top_3_purchases_df)

    with col2:
        st.subheader(':red[🎉 top 3 similarity scores products]')
        top_3_products_df = product_data_df.head(3)
        st.write(top_3_products_df)


    #############################################    
    # MESSAGE CHATTING
    ############################################### 
    message_objects = []
    message_objects.append({"role":"system", "content":"You're a chatbot helping customers with beauty-related questions and helping them with product recommendations"})
    message_objects.append({"role":"user", "content": customer_input})

    # create a string of the previous purchases from our top 3 purchases DataFrame:
    prev_purchases = ". ".join([f"{row['combined']}" for index, row in top_3_purchases_df.iterrows()])
    st.write(f"Top 3 Previous Purchase : {prev_purchases}")
    message_objects.append({"role":"user", "content": f"Here're my latest product orders: {prev_purchases}"})

    message_objects.append({"role":"user", "content": f"Please give me a detailed explanation of your recommendations"})
    message_objects.append({"role":"user", "content": "Please be friendly and talk to me like a person, don't just give me a list of recommendations"})

    message_objects.append({"role": "assistant", "content": f"I found these 3 products I would recommend"})

    products_list = []

    # add products table
    for index, row in top_3_products_df.iterrows():
        brand_dict = {'role': "assistant", "content": f"{row['combined']}"}
        products_list.append(brand_dict)
        # st.caption(brand_dict)

    # st.caption('Add product list to message')
    message_objects.extend(products_list)  

    message_objects.append({"role": "assistant", "content":"Here's my summarized recommendation of products, and why it would suit you:"})
    

    st.subheader(":blue[Step 6. 📃ChatML prompt:]")
    msg = ""
    for message in message_objects:
        msg = msg + message['content']
        # st.write(message['content'])

    completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = message_objects
            )
    st.warning(msg)
    st.subheader(":green[Step 7. ❤️ Recommendations]")
    st.info(completion.choices[0].message['content'])
    



