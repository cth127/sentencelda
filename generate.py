from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from pathlib import Path
import json, pickle


def write_json(res, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        f.close()


GPT_SIZE = {"-xl": 1600, "-large": 1280, "-medium": 1024, "": 768}
gpt_size = "-xl"
if gpt_size == "":
    multiplier = GPT_SIZE[gpt_size] // 768
else:
    multiplier = GPT_SIZE[gpt_size] // 768 + 1
encode_length = GPT_SIZE[gpt_size]


def feature(decoder, decoder_tokenizer):
    with torch.no_grad():
        model_path = Path(__file__).parents[0] / "params/table_means.pkl"
        decoder_embedding = decoder.get_input_embeddings()
        with open(model_path, "rb") as f:
            topic_mean = pickle.load(f)

        encoding = torch.tensor(topic_mean, dtype=torch.float32)
        encoding = encoding.repeat(1, multiplier)[:, :encode_length]
        encoding = encoding.unsqueeze(1)
        output_id = list()
        for _ in range(64):
            output = decoder(inputs_embeds=encoding).logits[:, -1, :].argmax(axis=1)
            output_id.append(output)
            new_emb = decoder_embedding(output).unsqueeze(1)
            encoding = torch.concat([encoding, new_emb], dim=1)
            if (output == 50256).all():
                break
        output_id = torch.stack(output_id).transpose(0, 1)
        ret = decoder_tokenizer.batch_decode(output_id, skip_special_tokens=True)
        result = {k: v.strip() for k, v in enumerate(ret)}
        write_json(result, "result.json")


if __name__ == "__main__":
    decoder_tokenizer = GPT2Tokenizer.from_pretrained(f"gpt2{gpt_size}")
    decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
    decoder = GPT2LMHeadModel.from_pretrained("cth127/gpt-xl-sentencebert-generation")
    feature(decoder, decoder_tokenizer)
