import os
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel, DistilBertTokenizer, DistilBertModel
from PIL import Image
from tqdm import tqdm

# Nastavitve
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model.eval()

# Funkcija za embedanje posamezne slike in naslova
def embed_example(image_path, title):
    try:
        image = Image.open(image_path).convert("RGB")
    except:
        image = Image.new("RGB", (224, 224))

    with torch.no_grad():
        image_inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
        image_embed = clip_model.get_image_features(**image_inputs).squeeze(0).cpu()

        text_inputs = bert_tokenizer(title, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        text_embed = bert_model(**text_inputs).last_hidden_state.mean(dim=1).squeeze(0).cpu()

    return image_embed, text_embed

# Glavna funkcija za celoten set
def process_dataset(tsv_path, image_folder, output_path):
    df = pd.read_csv(tsv_path, sep="\t")
    data = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(image_folder, row['id'] + '.jpg')
        title = row['title']
        label = int(row['2_way_label'])

        image_embed, text_embed = embed_example(img_path, title)
        data.append({
            'id': row['id'],
            'image_embed': image_embed,
            'text_embed': text_embed,
            'label': label
        })

    torch.save(data, output_path)
    print(f"Shranjeno v {output_path}")

# Uporaba
if __name__ == '__main__':
    process_dataset("multimodal_train.tsv", "images_test", "train_embeddings.pt")
    process_dataset("multimodal_validate.tsv", "images_validate", "val_embeddings.pt")
    process_dataset("multimodal_test_public.tsv", "images_test", "test_embeddings.pt")
