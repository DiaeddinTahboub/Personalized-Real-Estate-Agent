import gradio as gr
from setup import connect_to_lancedb
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema import HumanMessage

load_dotenv()


# Load CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
embedder = OpenAIEmbeddings()
schemas = [
    ResponseSchema(
        name="rankings", description="List of listings ranked with their descriptions"
    ),
    ResponseSchema(name="summary", description="Overall summary about the listings"),
]

parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
# Question list
questions = [
    "How big do you want your house to be?",
    "What are 3 most important things for you in choosing this property?",
    "Which amenities would you like?",
    "Which transportation options are important to you?",
    "How urban do you want your neighborhood to be?",
]

# App state
state = {"index": 0, "answers": []}

# Function to retrieve matches
def retrieve_best_matches(query, num_matches=5):
    db = connect_to_lancedb()
    listings_table = db.open_table("listings")
    images_table = db.open_table("images")

    # Split and embed lines with CLIP
    lines = [line.strip() for line in query.strip().split("\n") if line.strip()]
    embeddings = []
    for line in lines:
        inputs = processor(
            text=line, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            embedding = model.get_text_features(**inputs)
            embeddings.append(embedding)
    pooled_embedding = torch.mean(torch.stack(embeddings), dim=0)
    query_embedding_image_np = pooled_embedding.cpu().numpy()

    # Embed full query with OpenAI
    query_embedding_listing = np.array(embedder.embed_query(query))

    listings_results = (
        listings_table.search(query_embedding_listing).limit(num_matches).to_pandas()
    )
    images_results = (
        images_table.search(query_embedding_image_np).limit(num_matches).to_pandas()
    )

    return listings_results.loc[:, "listing"], images_results.loc[:, "image_path"]


def start_chat():

    state["index"] = 0
    state["answers"] = []
    return (
        questions[0],
        "",
        gr.update(visible=True),
        gr.update(visible=False),
        "",
        gr.update(visible=False),
    )


def submit_answer(answer):
    if answer.strip() == "" or answer == "Please enter an answer." or len(answer) < 3:
        return (
            questions[state["index"]],
            "Please enter an answer.",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            gr.update(visible=False),
        )

    state["answers"].append(answer.strip())
    state["index"] += 1

    if state["index"] < len(questions):
        return (
            questions[state["index"]],
            "",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            gr.update(visible=False),
        )
    else:
        return (
            "",
            "",
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            gr.update(visible=False),
        )


def done():
    final_answers = "\n".join(state["answers"])
    retrieved_listings, retrieved_images = retrieve_best_matches(final_answers)

    prompt = f"""
    You are a real estate agent. You are given a list of answers. You will use the answers and the retrieved listings to generate a report of why each listing is a good match for the answers.
    Here are the answers:
    {final_answers}
    Here are the listings:
    {retrieved_listings.to_list()}
    Please rank the listings from best to worst match for the answers and provide a rich summary each listing.
    like below:
    Listing 1:
    - Description: [description]
    - Why it's a good match: [reason]
    - Summary: [summary]
    """
    response = llm.invoke([HumanMessage(content=prompt)])

    response_text = response.content

    image_paths = retrieved_images.tolist()  # Ensure it's a list of strings
    return (
        response_text,
        gr.update(visible=False),
        gr.update(visible=True),
        image_paths,
        gr.update(visible=True),
    )


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🏡 Personalized Real Estate Finder")

    start_btn = gr.Button("Start")
    question_txt = gr.Textbox(label="Question", interactive=False, lines=2)
    answer_input = gr.Textbox(label="Your Answer", lines=2)
    submit_btn = gr.Button("Submit Answer", visible=False)
    done_btn = gr.Button("Done", visible=False)
    output = gr.Textbox(label="AI Response", interactive=False, lines=10)
    gallery = gr.Gallery(label="Matching Properties", visible=False)

    start_btn.click(
        start_chat,
        outputs=[question_txt, answer_input, submit_btn, done_btn, output, gallery],
    )
    submit_btn.click(
        submit_answer,
        inputs=answer_input,
        outputs=[question_txt, answer_input, submit_btn, done_btn, output, gallery],
    )
    done_btn.click(done, outputs=[output, done_btn, start_btn, gallery, gallery])

demo.launch()
