import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.embeddings import OpenAIEmbeddings
import numpy as np
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import lancedb
import pyarrow as pa
from dotenv import load_dotenv

load_dotenv()


def prepare_listings_table(db, chat_model):
    listings = [
        """Neighborhood: Green Oaks
    Price: $800,000
    Size: 2,500 sqft
    Bedrooms: 4
    Bathrooms: 3
    Description: This spacious modern home features an open floor plan, hardwood floors, and a gourmet kitchen with stainless steel appliances. The master suite includes a walk-in closet and spa-like bathroom. The backyard is perfect for entertaining with a patio and fire pit.
    Neighborhood Description: Green Oaks is a close-knit, environmentally-conscious community with access """
    ]

    def generate_listings(num_listings, listings):
        while len(listings) < num_listings:
            prompt = f"""I want you to generate 3 random, and very diverse listings for houses like this:
            {listings[-1]}
            Note: Separate each two listings with 5 hyphens (-----), and keep them short and varied in style.
            """
            response = chat_model.invoke([HumanMessage(content=prompt)])
            new_listings = response.content.split("-----")
            new_listings.pop()
            cleaned_listings = [listing.strip() for listing in new_listings]
            listings.extend(cleaned_listings)
        print("Finished generating listings")
        return listings

    listings = generate_listings(100, listings)

    embedder = OpenAIEmbeddings()

    embeddings = [np.array(embedder.embed_query(listing)) for listing in listings]

    df = pd.DataFrame({"listing": listings, "vector": embeddings})

    listing_embedding_size = len(df["vector"].iloc[0])

    listing_schema = pa.schema(
        [
            ("vector", pa.list_(pa.float32(), list_size=listing_embedding_size)),
            ("listing", pa.string()),
        ]
    )

    listings_table = db.create_table(
        "listings",
        data=df.to_dict(orient="records"),
        schema=listing_schema,
        mode="overwrite",
    )
    print("Listings table created")


def prepare_images_table(db, processor, model):
    image_paths = ["images/" + path for path in os.listdir("images")]

    images = [Image.open(path).convert("RGB") for path in image_paths]

    inputs = processor(images=images, return_tensors="pt", padding=True)

    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)

    image_embeddings_np = image_embeddings.cpu().numpy()

    df = pd.DataFrame({"image_path": image_paths, "vector": list(image_embeddings_np)})
    image_embedding_size = len(df["vector"].iloc[0])

    image_schema = pa.schema(
        [
            ("vector", pa.list_(pa.float32(), list_size=image_embedding_size)),
            ("image_path", pa.string()),
        ]
    )

    images_table = db.create_table(
        "images",
        data=df.to_dict(orient="records"),
        schema=image_schema,
        mode="overwrite",
    )


def connect_to_lancedb():
    db = lancedb.connect("my_lancedb")
    return db


def main():

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    chat_model = ChatOpenAI(model="gpt-3.5-turbo")
    db = connect_to_lancedb()

    prepare_listings_table(db, chat_model)
    prepare_images_table(db, processor, model)


if __name__ == "__main__":
    main()
