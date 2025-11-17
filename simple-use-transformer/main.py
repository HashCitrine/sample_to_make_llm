import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def simple_inference(input_text: str):
    model_name = "Qwen/Qwen3-1.7B"  # smaller model

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)

        device = torch.device(
            'cuda' if torch.cuda.is_available()  # nvidia gpu
            else "mps" if torch.backends.mps.is_available()  # silicon mac
            else 'cpu'
        )

        print(f"Using device: {device}")

        model.to(device)

        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        gen_params = {
            "max_length": 500,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "num_return_sequences": 1,
        }

        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], **gen_params)

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception as e:
        print("Error during model inference:", str(e))


if __name__ == "__main__":
    input_text = "최신 LLM 사례를 보여줘."
    output_text = simple_inference(input_text)
    print(output_text)
