import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
from typing import TypedDict
from dotenv import load_dotenv
import mysql.connector
from sentence_transformers import SentenceTransformer
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
import faiss
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import numpy as np



# Load environment variables
load_dotenv()

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_NAME = os.getenv('DB_NAME', 'init_db')

CHAT_MODEL = 'qwen2.5:7b-instruct-q4_K_M'

# --- Database Connection ---
def db_connect():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

# --- Category Fetching ---
def fetch_all_categories():
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT category FROM products")
    categories = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return categories

CATEGORIES = fetch_all_categories()
CATEGORY_LIST_STR = ", ".join(CATEGORIES)

# --- Embedding Setup (sentence-transformers) ---
embed_model = SentenceTransformer('thenlper/gte-large')

def get_embedding_local(text: str) -> list:
    return embed_model.encode(text).tolist()

# --- FAISS Indexing ---
def get_all_product_texts():
    conn = db_connect()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name, brand, category, description FROM products")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    docs = []
    for row in rows:
        text = f"{row['name']} {row['brand']} {row['category']} {row['description']}"
        docs.append(Document(page_content=text, metadata={"id": row["id"]}))
    return docs

# Load or Create FAISS index and doc store
index = None
if os.path.exists("product_index.faiss") and os.path.exists("doc_store.json"):
    index = faiss.read_index("product_index.faiss")
    with open("doc_store.json", "r", encoding="utf-8") as f:
        doc_store_data = json.load(f)
    doc_store = {
        int(k): Document(page_content=v["page_content"], metadata=v["metadata"])
        for k, v in doc_store_data.items()
    }
else:
    docs = get_all_product_texts()
    embeddings = [get_embedding_local(doc.page_content) for doc in docs]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, "product_index.faiss")

    doc_store = {i: doc for i, doc in enumerate(docs)}
    with open("doc_store.json", "w", encoding="utf-8") as f:
        json.dump({
            i: {"page_content": doc.page_content, "metadata": doc.metadata} for i, doc in doc_store.items()
        }, f, indent=2)

# --- Semantic Tool ---
@tool
def semantic_product_search(query: str) -> str:
    """Search for products semantically using vector embeddings."""
    print(f'Debug: Semantic search for query: {query}')
    if index.ntotal == 0:
        return "Product search unavailable."
    query_embedding = get_embedding_local(query)
    D, I = index.search(np.array([query_embedding]).astype("float32"), k=5)

    if not I[0].size:
        return "No products found for your search."

    return "\n".join(
        f"{doc_store[i].metadata['id']}. {doc_store[i].page_content}" for i in I[0]
    )

@tool
def get_all_categories() -> str:
    """Return all available product categories from the store."""
    print('Debug: Fetching all categories')
    return "Available categories:\n" + CATEGORY_LIST_STR

@tool
def search_products(query: str) -> str:
    """Search products by keyword in name, category or brand."""
    print(f'Debug: Searching products with query: {query}')
    conn = db_connect()
    cursor = conn.cursor(dictionary=True)
    sql = """
        SELECT id, name, brand, category, price FROM products 
        WHERE name LIKE %s OR category LIKE %s OR brand LIKE %s
        LIMIT 10
    """
    wildcard = f"%{query}%"
    cursor.execute(sql, (wildcard, wildcard, wildcard))
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    if not results:
        return f"No products found for '{query}'."

    return "\n".join(
        f"{p['id']}. {p['name']} - {p['brand']} - {p['category']} - ${p['price']:.2f}" for p in results
    )

@tool
def list_products_under(price: float, gender: str = '', last_category: str = '') -> str:
    """List products under a specific price. If a category was searched before, it will be used."""
    category = state.get('last_category', '').lower()

    print(f"Debug: Listing products under ${price}")
    if category:
        print(f"Debug: Filtering by category: {category}")
    else:
        print("Debug: No category context available.")

    conn = db_connect()
    cursor = conn.cursor(dictionary=True)
    user_gender = gender.lower()

    if category:
        sql = """
            SELECT id, name, brand, category, price 
            FROM products 
            WHERE price <= %s AND LOWER(category) LIKE %s 
            ORDER BY price ASC 
            LIMIT 10
        """
        cursor.execute(sql, (price, f"%{category}%"))
    else:
        sql = """
            SELECT id, name, brand, category, price 
            FROM products 
            WHERE price <= %s 
            ORDER BY price ASC 
            LIMIT 10
        """
        cursor.execute(sql, (price,))

    results = cursor.fetchall()
    cursor.close()
    conn.close()

    if not results:
        return f"No products found under ${price}."

    return "\n".join(
        f"{p['id']}. {p['name']} by {p['brand']} - {p['category']} - ${p['price']:.2f}" for p in results
    )




@tool
def get_product_details(product_name: str) -> str:
    """Get full details of a product by its name."""
    print(f'Debug: Getting product details for {product_name}')
    conn = db_connect()
    cursor = conn.cursor(dictionary=True)
    sql = "SELECT * FROM products WHERE name LIKE %s LIMIT 1"
    cursor.execute(sql, (f"%{product_name}%",))
    product = cursor.fetchone()
    cursor.close()
    conn.close()

    if not product:
        return f"No product found with name '{product_name}'."

    return (
        f"Product Details:\n"
        f"ID: {product['id']}\n"
        f"Name: {product['name']}\n"
        f"Brand: {product['brand']}\n"
        f"Category: {product['category']}\n"
        f"Price: ${product['price']:.2f}\n"
        f"Description: {product['description']}"
    )

@tool
def recommend_similar_products(product_id: int) -> str:
    """Recommend other products from the same category as the given product ID."""
    print(f'Debug: Recommending similar products for product ID')
    conn = db_connect()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT category FROM products WHERE id = %s", (product_id,))
    result = cursor.fetchone()

    if not result:
        cursor.close()
        conn.close()
        return f"No product found with ID {product_id}."

    category = result['category']
    cursor.execute(
        "SELECT id, name, brand, price FROM products WHERE category = %s AND id != %s LIMIT 5",
        (category, product_id)
    )
    recommendations = cursor.fetchall()
    cursor.close()
    conn.close()

    if not recommendations:
        return f"No similar products found for product ID {product_id}."

    return (
        f"Products similar to item {product_id} (category: {category}):\n" +
        "\n".join(
            f"{p['id']}. {p['name']} - {p['brand']} - ${p['price']:.2f}" for p in recommendations
        )
    )

@tool
def update_user_profile(name: str = "", gender: str = "") -> str:
    """Update the user's profile based on their name or gender (e.g., 'Maria' implies female,'Joe' implies male)."""
    print('Debug: Updating user profile with')
    updates = []
    if name:
        state['user_profile']['name'] = name
        updates.append(f"Name set to {name}")
    if gender:
        gender = gender.lower()
        state['user_profile']['gender'] = gender
        updates.append(f"Gender set to {gender}")
    return " | ".join(updates) if updates else "No profile updates were made."

@tool
def list_products_by_category(category: str, gender: str = '') -> str:
    """
    List products in a specific category. 
    If the category is gender-neutral (e.g. "shoes", "shirt"), and the user's gender is known,
    use that to pick the correct category like "women shoes" or "men shirts".
    the category should be one of the existing categories.(- mens-shoes
- beauty
- laptops
- mens-shirts
- mobile accessories
- smartphones
- tablets
- vehicle
- womens-shoes
- tops
- sports accessories
- skin care)
    """
    print(f'Debug: Listing products by category:{category}')
    user_gender = gender.lower()
    category_clean = category.strip().lower()

    # Try to adjust based on gender if applicable
    if user_gender:
        gendered_category = f"{user_gender} {category_clean}"
        matching_categories = [cat.lower() for cat in CATEGORIES]
        if gendered_category in matching_categories:
            category = gendered_category
    print(f'Debug: Final category used for listing: {category}')
    conn = db_connect()
    cursor = conn.cursor(dictionary=True)

    sql = "SELECT id, name, brand, price FROM products WHERE LOWER(category) LIKE %s LIMIT 10"
    cursor.execute(sql, (f"%{category.lower()}%",))
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    if not results:
        return f"No products found in the category '{category}'."

    return "\n".join(
        f"{p['id']}. {p['name']} - {p['brand']} - ${p['price']:.2f}" for p in results
    )


@tool
def get_product_images(product_name: str) -> str:
    """Get image URLs of a product by its name."""
    print(f'Debug: Getting images for product {product_name}')
    conn = db_connect()
    cursor = conn.cursor(dictionary=True)

    sql = "SELECT images FROM products WHERE name LIKE %s LIMIT 1"
    cursor.execute(sql, (f"%{product_name}%",))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if not result or not result.get('images'):
        return f"No images found for product '{product_name}'."

    # Parse images assuming JSON string
    import json
    try:
        images = json.loads(result['images'])
    except Exception:
        return "Images data format error."

    if not images:
        return f"No images found for product '{product_name}'."

    # Format nicely
    formatted_links = "\n".join(f"- {url}" for url in images)
    return f"Image URLs for '{product_name}':\n{formatted_links}"

# -------------------- Chat Model + Graph --------------------

llm = init_chat_model(CHAT_MODEL, model_provider='ollama')

llm = llm.bind_tools([
    semantic_product_search,
    get_all_categories,
    search_products,
    list_products_under,
    list_products_by_category,
    get_product_details,
    recommend_similar_products,
    get_product_images, 
    update_user_profile
])

raw_llm = init_chat_model(CHAT_MODEL, model_provider='ollama')

class ChatState(TypedDict):
    messages: list
    user_profile: dict 
    last_category: str

def llm_node(state):
    response = llm.invoke(state['messages'])
    return {'messages': state['messages'] + [response]}

def router(state):
    last_message = state['messages'][-1]
    return 'tools' if getattr(last_message, 'tool_calls', None) else 'end'

tool_node = ToolNode([
    semantic_product_search,
    get_all_categories,
    search_products,
    list_products_under,
    list_products_by_category,
    get_product_details,
    recommend_similar_products,
    get_product_images,
    update_user_profile
])

def tools_node(state):
    result = tool_node.invoke(state)
    updated_messages = state['messages'] + result['messages']
    last_tool_call = result['messages'][-1].tool_calls[0] if hasattr(result['messages'][-1], 'tool_calls') else None

    # Track last_category if the category listing tool was used
    last_category = state.get('last_category', '')

    if last_tool_call and last_tool_call.name == 'list_products_by_category':
        # Extract the category argument (it should be passed as a JSON string)
        try:
            tool_args = json.loads(last_tool_call.args)
            last_category = tool_args.get('category', last_category).lower()
            print(f"Debug: Stored last_category → {last_category}")
        except Exception as e:
            print("Warning: Failed to parse tool args for last_category:", e)

    return {
        'messages': updated_messages,
        'user_profile': state.get('user_profile', {}),
        'last_category': last_category
    }



builder = StateGraph(ChatState)
builder.add_node('llm', llm_node)
builder.add_node('tools', tools_node)
builder.add_edge(START, 'llm')
builder.add_edge('tools', 'llm')
builder.add_conditional_edges('llm', router, {'tools': 'tools', 'end': END})

graph = builder.compile()
compiled_app = graph  # 👈 This makes it importable in your Streamlit app

# -------------------- Chat Loop --------------------
if __name__ == '__main__':
    state = {
        'messages': [
            {
                'role': 'system',
                'content': (
                    "You are a helpful and knowledgeable e-commerce assistant for an online store. "
                    "Your goal is to help customers find the right products based on their needs. "
                    "If a user asks to browse or search by category, first call the get_all_categories tool "
                    "to retrieve the full list of available categories. Then, identify the most relevant category "
                    "based on the user's request and use the list_products_by_category tool to show matching products. "
                    "Only use existing categories. Avoid guessing or making up categories that are not listed."
                    "Never invent product names or list items that were not provided by the tools. "
                    "If there are only 3 or 5 results, list only those — do not make up additional ones."
                    "You should extract the user's name or gender from conversation. If the user says something like "
                    "'my name is Maria' or 'I’m a woman', call the update_user_profile tool. "
                    "When the user asks for 'shoes', prefer gender-specific categories (like 'women shoes') "
                    "if the gender is known in the user_profile. "
                    "Remember user preferences such as name, gender, and recently browsed categories. "
                    "Use this information to personalize product recommendations. "
                    "If the user previously searched a category, use that category when filtering by price. "
                    "If the use ask for images, use the get_product_images tool to retrieve product images. "
                    "When the user describes a product using natural language, feelings, qualities, or general descriptions — and not exact names, categories, or brands — use the semantic_product_search tool.Examples include: something durable for travel,a powerful laptop or a product good for sensitive skin."
                    "Don't tell the user about the tools you are using and how you think."
                )
            }
        ],
    'user_profile': {},
    'last_category': ''
    }

    print('🛍️ E-Commerce Chatbot ready! Ask your questions.\n')

    while True:
        user_message = input('> ')
        if user_message.lower() == 'quit':
            break
        state['messages'].append({'role': 'user', 'content': user_message})
        state = graph.invoke(state)
        print(state['messages'][-1].content, '\n')